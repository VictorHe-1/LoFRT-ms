import math

import mindspore as ms
from mindspore import nn, ops


def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: bool = True, dtype = ms.float32) -> ms.Tensor:
    """Generates a coordinate grid for an image.

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (bool): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        ms.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    xs = ops.arange(width, dtype=dtype)
    ys = ops.arange(height, dtype=dtype)

    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2

    # generate grid by stacking coordinates
    base_grid = ops.stack(ops.meshgrid(xs, ys, indexing='xy'))  # (2, h, w)
    return base_grid.permute(1, 2, 0)  # (h, w, 2)


def spatial_expectation2d(
        input: ms.Tensor,
        grid: ms.Tensor,
) -> ms.Tensor:
    r"""Computes the expectation of coordinate values using spatial probabilities.

    The input heatmap is assumed to represent a valid spatial probability
    distribution, which can be achieved using
    :class:`~kornia.geometry.dsnt.spatial_softmax2d`.

    Returns the expected value of the 2D coordinates.
    The output order of the coordinates is (x, y).

    Arguments:
        input (ms.Tensor): the input tensor representing dense spatial probabilities.
        normalized_coordinates (bool): whether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples:
        >>> heatmaps = ms.Tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 0.]]]])
        >>> coords = spatial_expectation2d(heatmaps, False)
        tensor([[[1.0000, 2.0000]]])
    """

    batch_size, channels = input.shape[:2]

    pos_x = grid[..., 0].reshape(-1)
    pos_y = grid[..., 1].reshape(-1)

    input_flat = input.view(batch_size, channels, -1)

    # Compute the expectation of the coordinates.
    expected_x = ops.sum(pos_x * input_flat, -1, keepdim=True)
    expected_y = ops.sum(pos_y * input_flat, -1, keepdim=True)

    output = ops.cat([expected_x, expected_y], -1)

    return output.view(batch_size, channels, 2)  # (bs, c, 2)


class FineMatching(nn.Cell):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        super().__init__()

    def construct(self, feat_f0, feat_f1, match_kpts_c0, match_kpts_c1, hw_i0, hw_f0, scale_1):
        """
        Fine-level matching procedure that refines the matches found in the coarse-level descriptor.

        Args:
            feat_f0 (ms.Tensor): Shape [bs, num_match, ww, c]. The fine-level feature descriptor of image 0.
            feat_f1 (ms.Tensor): Shape [bs, num_match, ww, c]. The fine-level feature descriptor of image 1.
            match_kpts_c0 (ms.Tensor): The matched keypoints in image 0 from the coarse-level descriptor.
            match_kpts_c1 (ms.Tensor): The matched keypoints in image 1 from the coarse-level descriptor.
            hw_i0 (tuple): Height and width of the original image 0.
            hw_f0 (tuple): Height and width of the fine feature map from image 0.
            scale_1 (ms.Tensor): The scaling applied to image 1 during preprocessing.

        Returns:
            Tuple(Tensor):
            - match_kpts_f0 (ms.Tensor): The refined points in image 0 after fine-level processing.
            - match_kpts_f1 (ms.Tensor): The refined points in image 1 after fine-level processing.
            - expec_f (ms.Tensor): The distribution (mean, std) for fine-level supervision.
        """
        bs, num_match, ww, c = feat_f0.shape
        w = int(math.sqrt(ww))
        scale = hw_i0[0] / hw_f0[0]

        feat_f0_center = feat_f0[:, :, ww//2, :]  # (bs, num_match, c)
        # (bs, num_match, ww, c) * (bs, num_match, c, 1) -> (bs, num_match, ww, 1) -> (bs, num_match, ww)
        sim_matrix = ops.bmm(feat_f1, feat_f0_center.unsqueeze(-1)).squeeze(-1)

        softmax_temp = 1. / c**.5  # the higher, the less concentrated
        heatmap = ops.softmax(softmax_temp * sim_matrix, axis=2)

        # compute coordinates from heatmap
        grid_normalized = create_meshgrid(w, w, True, heatmap.dtype).reshape(-1, 2)  # (ww, 2)
        coords_normalized = spatial_expectation2d(heatmap, grid_normalized)  # (bs, num_match, 2) normed to -1, 1

        # compute std over <x, y>, equals to weighted_avg(x-x_bar)
        # (ww, 2) * (bs, num_match, ww, 1) -> (bs, num_match, ww, 2) sum2 -> (bs, num_match, 2)
        var = ops.sum(grid_normalized**2 * heatmap.view(bs, num_match, ww, 1), dim=2) - coords_normalized**2
        std = ops.sum(ops.sqrt(ops.clamp(var, min=1e-10)), -1)  # (bs, num_match)  clamp needed for numerical stability
        
        # distribution (mean, std) for fine-level supervision
        expec_f = ops.cat([coords_normalized, std.unsqueeze(-1)], -1)  # (bs, num_match, 2)

        # compute absolute kpt coords
        match_kpts_f1 = match_kpts_c1 + (coords_normalized * (w // 2) * scale * scale_1[0])  # (bs, num_match, 2) TODO check length when training

        match_kpts_f0 = ops.stop_gradient(match_kpts_c0)  # the key points of the first image remains unchanged
        match_kpts_f1 = ops.stop_gradient(match_kpts_f1)

        return match_kpts_f0, match_kpts_f1, expec_f
