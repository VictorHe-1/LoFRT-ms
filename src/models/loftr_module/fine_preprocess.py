import mindspore as ms
from mindspore import nn, ops
import mindspore.common.initializer as init

class FinePreprocess(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine_window_size']

        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Dense(d_model_c, d_model_f, has_bias=True)
            self.merge_feat = nn.Dense(2*d_model_f, d_model_f, has_bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.get_parameters():
            if p.dim() > 1:
                p.set_data(init.initializer(init.HeNormal(mode='fan_out', nonlinearity='relu'), p.shape, p.dtype))

    def construct(self, feat_f0, feat_f1, feat_c0, feat_c1, hw_c0, hw_f0, coarse_match_ids):
        W = self.W
        stride = hw_f0[0] // hw_c0[0]

        # Step1. unfold(crop) all local windows
        feat_f0_unfold = ops.unfold(feat_f0, kernel_size=(self.W, self.W), stride=stride, padding=self.W//2)  # TODO to reshape and concat
        n, c_ww, l = feat_f0_unfold.shape
        _, c, _, _ = feat_f0.shape
        # feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
        feat_f0_unfold = feat_f0_unfold.reshape(n, c, self.W**2, l).transpose(0, 3, 2, 1)

        feat_f1_unfold = ops.unfold(feat_f1, kernel_size=(self.W, self.W), stride=stride, padding=self.W//2)
        n, c_ww, s = feat_f1_unfold.shape
        _, c, _, _ = feat_f1.shape
        feat_f1_unfold = feat_f1_unfold.reshape(n, c, self.W**2, s).transpose(0, 3, 2, 1)  # bs, c, ww, s -> bs, s, ww, c

        # Step2. gather features for fine matching according to coarse_match
        # (bs, l/s,, c) -> (bs, num_coarse_match, c)
        _, num_coarse_match, _ = coarse_match_ids.shape
        feat_c0 = ops.gather(feat_c0, coarse_match_ids[..., 0], axis=1, batch_dims=1)
        feat_c1 = ops.gather(feat_c1, coarse_match_ids[..., 1], axis=1, batch_dims=1)
        # (bs, l/s, ww, c) -> (bs, num_coarse_match, ww, c)
        feat_f0_unfold = ops.gather(feat_f0_unfold, coarse_match_ids[..., 0], axis=1, batch_dims=1)
        feat_f1_unfold = ops.gather(feat_f1_unfold, coarse_match_ids[..., 1], axis=1, batch_dims=1)

        # use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(ops.cat([feat_c0, feat_c1], axis=1))  # (bs, 2*num_coarse_match, c)
            fine = ops.cat([feat_f0_unfold, feat_f1_unfold], axis=1)  # (bs, 2*num_coarse_match, ww, c)
            coarse = ops.repeat_elements(ops.expand_dims(feat_c_win, axis=2), rep=W**2, axis=2)  # (bs, 2*num_coarse_match, ww, c)
            feat_cf_win = self.merge_feat(ops.cat([fine, coarse], axis=-1))  # (bs, 2*num_coarse_match, ww, 2c) -> (bs, 2*num_coarse_match, ww, c)
            feat_f0_unfold, feat_f1_unfold = ops.split(feat_cf_win, num_coarse_match, axis=1)  # (bs, num_coarse_match, ww, 2c)

        return feat_f0_unfold, feat_f1_unfold
