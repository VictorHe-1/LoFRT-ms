import logging
from math import log

import mindspore as ms
from mindspore import ops, Tensor

from .geometry import warp_kpts

_logger = logging.getLogger(__name__)


##############  ↓  Coarse-Level supervision  ↓  ##############

def repeat(tensor, pattern, c):
    tensor = tensor.view(tensor.shape[0], -1)
    tensor = tensor.unsqueeze(-1)
    tensor = tensor.tile((1, 1, c))
    return tensor


def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: bool = True
) -> Tensor:
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
    xs: Tensor = ops.linspace(0, width - 1, width)
    ys: Tensor = ops.linspace(0, height - 1, height)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: Tensor = ops.stack(ops.meshgrid(xs, ys, indexing="ij"), axis=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2


def spvs_coarse(data, config, data_cols):
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
    data_cols.index("image0")
    N, _, H0, W0 = data[data_cols.index("image0")].shape
    _, _, H1, W1 = data[data_cols.index("image1")].shape
    scale = config['LOFTR']['RESOLUTION'][0]
    scale0 = scale * data[data_cols.index("scale0")][:, None] if 'scale0' in data_cols else scale
    scale1 = scale * data[data_cols.index("scale1")][:, None] if 'scale1' in data_cols else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = ops.stop_gradient(create_meshgrid(h0, w0, False).reshape(1, h0 * w0, 2).tile((N, 1, 1)))  # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = ops.stop_gradient(create_meshgrid(h1, w1, False).reshape(1, h1 * w1, 2).tile((N, 1, 1)))
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = ops.stop_gradient(mask_pts_at_padded_regions(grid_pt0_i, data[data_cols.index("mask0")]))
        grid_pt1_i = ops.stop_gradient(mask_pts_at_padded_regions(grid_pt1_i, data[data_cols.index("mask1")]))

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = ops.stop_gradient(warp_kpts(grid_pt0_i,
                                             data[data_cols.index('depth0')],
                                             data[data_cols.index('depth1')],
                                             data[data_cols.index('T_0to1')],
                                             data[data_cols.index('K0')],
                                             data[data_cols.index('K1')]))
    _, w_pt1_i = ops.stop_gradient(warp_kpts(grid_pt1_i,
                                             data[data_cols.index('depth1')],
                                             data[data_cols.index('depth0')],
                                             data[data_cols.index('T_1to0')],
                                             data[data_cols.index('K1')],
                                             data[data_cols.index('K0')]))
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)

    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = ops.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], axis=0)
    correct_0to1 = loop_back == ops.arange(h0 * w0)[None].tile((N, 1))
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = ops.zeros((N, h0 * w0, h1 * w1))
    b_ids, i_ids = ops.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    conf_matrix_gt = ops.stop_gradient(conf_matrix_gt)
    data.append(conf_matrix_gt)  # TODO: confirm data type
    data_cols.append("conf_matrix_gt")
    # data.update({'conf_matrix_gt': conf_matrix_gt})


    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        data_pair = (data[data_cols.index('pair_names_0')], data[data_cols.index('pair_names_1')])
        _logger.warning(f"No groundtruth coarse match found for: {data_pair}")
        # this won't affect fine-level loss calculation
        b_ids = Tensor([0], dtype=ms.int32)
        i_ids = Tensor([0], dtype=ms.int32)
        j_ids = Tensor([0], dtype=ms.int32)

    data_cols.extend(['spv_b_ids', 'spv_i_ids', 'spv_j_ids', 'spv_w_pt0_i', 'spv_pt1_i'])
    data.extend([b_ids, i_ids, j_ids, w_pt0_i, grid_pt1_i])
    return data, data_cols


def compute_supervision_coarse(data, config, data_cols):
    data_source = data[data_cols.index("dataset_name")]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_coarse(data, config, data_cols)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


##############  ↓  Fine-Level supervision  ↓  ##############

def spvs_fine(data, config, data_cols):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i, pt1_i = data[data_cols.index('spv_w_pt0_i')], data[data_cols.index('spv_pt1_i')]
    scale = config['LOFTR']['RESOLUTION'][1]
    radius = config['LOFTR']['FINE_WINDOW_SIZE'] // 2

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data[data_cols.index('b_ids')], data[data_cols.index('i_ids')], data[data_cols.index('j_ids')]

    # 3. compute gt
    scale = scale * data[data_cols.index('scale1')][b_ids] if 'scale0' in data_cols else scale
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius  # [M, 2]
    expec_f_gt = ops.stop_gradient(expec_f_gt)
    data_cols.append("expec_f_gt")
    data.append(expec_f_gt)
    return data, data_cols


def compute_supervision_fine(data, config, data_cols):
    data_source = data[data_cols.index("dataset_name")]
    if data_source.lower() in ['scannet', 'megadepth']:
        return spvs_fine(data, config, data_cols)
    else:
        raise NotImplementedError
