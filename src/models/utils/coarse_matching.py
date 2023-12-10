import mindspore as ms
from mindspore import nn, ops

INF = 1e9

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def back_mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1).int(), p_m0.sum(-1).max(-1).int()
    h1s, w1s = p_m1.sum(1).max(-1).int(), p_m1.sum(-1).max(-1).int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def mask_border_with_padding(m, bd, p_m0, p_m1):
    if bd <= 0:
        return

    p_m0 = p_m0.astype(ms.int32)  # (bs, h, w)
    p_m1 = p_m1.astype(ms.int32)
    bs = p_m0.shape[0]

    # (bs, 1, 1)  (bs, 1, 1)
    w0s, h0s  = p_m0[:, 0].sum().view(bs, 1, 1), p_m0[:, :, 0].sum().view(bs, 1, 1)
    w1s, h1s = p_m1[:, 0].sum().view(bs, 1, 1), p_m1[:, :, 0].sum().view(bs, 1, 1)

    h0_mask = ops.logical_and(ops.cumsum(p_m0, axis=1) <= h0s - bd, ops.cumsum(p_m0, axis=1) > bd)
    w0_mask = ops.logical_and(ops.cumsum(p_m0, axis=2) <= w0s - bd, ops.cumsum(p_m0, axis=2) > bd)
    hw0_mask = ops.logical_and(h0_mask, w0_mask)  # (bs, h0, w0)
    hw0_mask = ops.logical_and(hw0_mask, p_m0.astype(ms.bool_))

    h1_mask = ops.logical_and(ops.cumsum(p_m1, axis=1) <= h1s - bd, ops.cumsum(p_m1, axis=1) > bd)
    w1_mask = ops.logical_and(ops.cumsum(p_m1, axis=2) <= w1s - bd, ops.cumsum(p_m1, axis=2) > bd)
    hw1_mask = ops.logical_and(h1_mask, w1_mask)  # (bs, h1, w1)
    hw1_mask = ops.logical_and(hw1_mask, p_m1.astype(ms.bool_))

    mask_border = ops.logical_and(hw0_mask[:, :, :, None, None], hw1_mask[:, None, None, :, :])
    m = ops.logical_and(m, mask_border)

    return m


class CoarseMatching(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

        self.num_max_match = config.get('num_max_match', None)
        self.bmm = ops.BatchMatMul(transpose_b=True)

        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        if self.match_type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        else:
            raise NotImplementedError()

    def construct(self, feat_c0, feat_c1, hw_c0, hw_c1, hw_i0, hw_i1, mask_c0, mask_c1):
        """
        Args:
            feat0 (ms.Tensor): [N, L, C]
            feat1 (ms.Tensor): [N, S, C]
            data (dict)
            mask_i0_flat (ms.Tensor): [N, L] (optional), 0 for pad area
            mask_i1_flat (ms.Tensor): [N, S] (optional)
            mask_c0_margin (ms.Tensor): [N, h0, w0], compared to mask_i0_flat, only mask the margin with 0, this is a
                                       hack implementation of 'mask_border_with_padding'
            mask_c1_margin: see mask_c0_margin
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.shape[0], feat_c0.shape[1], feat_c1.shape[1], feat_c0.shape[2]

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5, [feat_c0, feat_c1])

        if self.match_type == 'dual_softmax':
            sim_matrix = self.bmm(feat_c0, feat_c1) / self.temperature
            mask_matrix = (mask_c0[..., None].astype(ms.int32) * mask_c1[:, None].astype(ms.int32)).bool()
            sim_matrix.masked_fill(~mask_matrix, -INF)
            conf_matrix = ops.softmax(sim_matrix, 1) * ops.softmax(sim_matrix, 2)  # dual softmax

        else:
            raise NotImplementedError()

        # predict coarse matches from conf_matrix
        # TODO check no grad ?equals to stop_gradient
        coarse_matches = self.get_coarse_match(conf_matrix, hw_c0, hw_c1, hw_i0, hw_i1, mask_c0, mask_c1,)
        return coarse_matches

    def get_coarse_match(self, conf_matrix, hw_c0, hw_c1, hw_i0, hw_i1, mask_c0, mask_c1):
        """
        Args:
            conf_matrix (ms.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:

        """
        bs, l, s = conf_matrix.shape
        # 1. confidence thresholding
        mask = conf_matrix > self.thr

        # 2. safe margin
        mask = mask.view(bs, hw_c0[0], hw_c0[1], hw_c1[0], hw_c1[1])
        mask = mask_border_with_padding(mask, self.border_rm,
                                 mask_c0.view(bs, hw_c0[0], hw_c0[1]),
                                 mask_c1.view(bs, hw_c1[0], hw_c1[1])) # mask_c0, 0 for pad area
        # mask_border = ops.logical_and(mask_c0[:, :, :, None, None], mask_c1[:, None, None, :, :])
        # mask = ops.logical_and(mask, mask_border)
        mask = mask.view(bs, l, s)

        # 3. mutual nearest
        mask = mask.astype(ms.int32)
        mask = mask \
            * (conf_matrix == ops.max(conf_matrix, axis=2, keepdims=True)[0]).astype(ms.int32) \
            * (conf_matrix == ops.max(conf_matrix, axis=1, keepdims=True)[0]).astype(ms.int32)

        # 4. find all valid coarse matches, Note that this only works when at most one `True` in each row
        mask_v, colum_ids = mask.max(2, return_indices=True)  # (bs, l)
        valids = ops.arange(l, dtype=ms.int32)
        invalids = ops.ones_like(valids) * l
        row_ids = ops.where(mask_v.astype(ms.bool_), valids, invalids) # (bs, l)
        # move the valid match to the front
        index = ops.argsort(row_ids.astype(ms.float32), axis=-1, descending=False)
        row_ids = ops.gather_elements(row_ids, dim=1, index=index)
        colum_ids = ops.gather_elements(colum_ids, dim=1, index=index)
        match_masks = row_ids != l

        # replace valid index to 0
        match_ids = ops.stack([row_ids % l, colum_ids], axis=-1)  # (bs, l, 2)
        # match_conf = ops.grid_sample(conf_matrix, match_ids, mode='nearest')  # (bs, l)

        # conf_rows = conf_matrix[row_ids]  # [bs, l, s]
        conf_rows = ops.gather(conf_matrix, input_indices=row_ids, axis=1, batch_dims=1)
        match_conf = ops.gather_elements(conf_rows, dim=2, index=colum_ids.expand_dims(-1))[..., 0]


        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # TODO add logic here
            pass

        # 4. Update with matches in original image resolution
        # scale = data['hw0_i'][0] / data['hw0_c'][0]
        # scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale  # TODO check data[‘scale0’]
        # scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        scale0 = scale1 = hw_i0[0] / hw_c0[0]
        mkpts_c0 = ops.stack([row_ids % hw_c0[1], row_ids // hw_c0[1]], axis=2) * scale0  # (bs, l, 2) in (w, h) format
        mkpts_c1 = ops.stack([colum_ids % hw_c1[1], colum_ids // hw_c1[1]], axis=2) * scale1

        # These matches is the current prediction (for visualization)
        # coarse_matches.update({
        #     'gt_mask': mconf == 0,
        #     'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
        #     'mkpts0_c': mkpts0_c[mconf != 0],
        #     'mkpts1_c': mkpts1_c[mconf != 0],
        #     'mconf': mconf[mconf != 0]
        # })

        if self.num_max_match is not None:
            match_masks = match_masks[:, :self.num_max_match]
            match_ids = match_ids[:, :self.num_max_match]
            match_conf = match_conf[:, :self.num_max_match]
            mkpts_c0 = mkpts_c0[:, :self.num_max_match]
            mkpts_c1 = mkpts_c1[:, :self.num_max_match]

        return ops.stop_gradient(match_ids), \
               ops.stop_gradient(match_masks), \
               ops.stop_gradient(match_conf), \
               ops.stop_gradient(mkpts_c0),\
               ops.stop_gradient(mkpts_c1)

