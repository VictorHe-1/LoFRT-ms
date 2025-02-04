import math
from mindspore import nn, ops


class PositionEncodingSine(nn.Cell):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = ops.zeros((d_model, *max_shape))
        y_position = ops.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = ops.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = ops.exp(ops.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = ops.exp(ops.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = ops.sin(x_position * div_term)
        pe[1::4, :, :] = ops.cos(x_position * div_term)
        pe[2::4, :, :] = ops.sin(y_position * div_term)
        pe[3::4, :, :] = ops.cos(y_position * div_term)
        self.pe = pe  # [C, H, W]

    def construct(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :x.shape[2], :x.shape[3]]
