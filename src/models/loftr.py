import mindspore as ms
from mindspore import nn, ops
from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


class LoFTR(nn.Cell):
    def __init__(self, config, loss=None):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()
        self.loss = loss  # only needed for training

        # For compute supervision: spvs_fine
        self.scale_spvs = config['resolution'][1]
        self.radius_spvs = config['fine_window_size'] // 2

    def compute_supervision_fine(self, scale1, spv_w_pt0_i, spv_pt1_i, match_ids):
        w_pt0_i, pt1_i = spv_w_pt0_i, spv_pt1_i
        scale = self.scale_spvs
        radius = self.radius_spvs

        # 2. get coarse prediction
        # match_ids: [bs, l, 2]
        i_ids, j_ids = match_ids[0][:, 0], match_ids[0][:, 1]
        b_ids = ms.Tensor([0 for _ in range(match_ids.shape[1])], dtype=ms.int32)

        # 3. compute gt
        scale = scale * scale1[0]
        # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
        # w_pt0_i: (1, 6400, 2) pt1_i: (1, 6400, 2) b_ids: (6400,)
        # expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius  # [M, 2]
        expec_f_gt = (ops.gather(w_pt0_i, i_ids, axis=1).squeeze(0) - ops.gather(pt1_i, j_ids, axis=1).squeeze(0)) / scale / radius
        return ops.stop_gradient(expec_f_gt)

    def construct(self,
                  img0,
                  img1,
                  mask_c0,
                  mask_c1,
                  scale0,
                  scale1,
                  conf_matrix_gt,
                  spv_w_pt0_i,
                  spv_pt1_i
                  ):
        """ 
        forward pass
        Args:
            image0: (ms.Tensor): (bs, 1, H, W)
            image1: (ms.Tensor): (bs, 1, H, W)
            mask_c0: (ms.Tensor[bool]): (bs, H, W) False indicates a padded position
            mask_c1: (ms.Tensor): (bs, H, W)
            scale0: (ms.Tensor): (bs, 2)
            scale1: (ms.Tensor): (bs, 2)
            # For training:
            spv_b_ids, spv_i_ids, spv_j_ids
        """
        bs = img0.shape[0]
        hw_i0, hw_i1 = img0.shape[2:], img1.shape[2:]  # initial spatial shape of the image pair

        # Step1: extract feature
        # image pairs are of the same shape, pad them in batch to speed up
        if hw_i0 == hw_i1:
            feats_c, feats_f = self.backbone(ops.cat([img0, img1], axis=0))
            feat_c0, feat_c1 = feats_c.split(bs)
            feat_f0, feat_f1 = feats_f.split(bs)
        else:
            feat_c0, feat_f0 = self.backbone(img0)
            feat_c1, feat_f1 = self.backbone(img1)

        hw_c0, hw_c1 = feat_c0.shape[2:], feat_c1.shape[2:]
        hw_f0, hw_f1 = feat_f0.shape[2:], feat_f1.shape[2:]

        # Step2: coarse-level self- and cross- attention
        feat_c0 = self.pos_encoding(feat_c0).flatten(start_dim=-2).swapaxes(1, 2)  # (bs, c, h, w) -> (bs, hw, c)
        feat_c1 = self.pos_encoding(feat_c1).flatten(start_dim=-2).swapaxes(1, 2)

        # padding mask, 0 for pad area
        mask_c0_flat, mask_c1_flat = mask_c0.flatten(start_dim=-2), mask_c1.flatten(start_dim=-2)  # (bs, c, hw)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0_flat, mask_c1_flat)
        # Step3: match coarse-level
        match_ids, match_masks, match_conf, match_kpts_c0, match_kpts_c1, conf_matrix = self.coarse_matching(feat_c0,
                                                                                                            feat_c1,
                                                                                                            hw_c0,
                                                                                                            hw_c1,
                                                                                                            hw_i0,
                                                                                                            hw_i1,
                                                                                                            mask_c0_flat,
                                                                                                            mask_c1_flat,
                                                                                                            scale0,
                                                                                                            scale1)

        # Step4: crop small patch of fine-feature-map centered at coarse feature map points
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, hw_c0, hw_f0,
                                                              match_ids)

        # Step4: fine-level self- and cross- attention
        feat_f0_unfold, feat_f1_unfold = self.loftr_fine_with_reshape(feat_f0_unfold, feat_f1_unfold)

        # Step5: match fine-level
        match_kpts_f0, match_kpts_f1, expec_f = self.fine_matching(feat_f0_unfold, feat_f1_unfold,
                                                                              match_kpts_c0, match_kpts_c1,
                                                                              hw_i0, hw_f0, scale1)
        if self.training:
            expec_f_gt = self.compute_supervision_fine(scale1, spv_w_pt0_i, spv_pt1_i, match_ids)
            return self.loss(expec_f,
                             expec_f_gt,
                             mask_c0,
                             mask_c1,
                             conf_matrix,
                             conf_matrix_gt,
                             match_masks)
        else:
            return match_kpts_f0, match_kpts_f1, match_conf, match_masks

    def loftr_fine_with_reshape(self, feat_f0_unfold, feat_f1_unfold):
        bs, num_coarse_match, ww, c = feat_f0_unfold.shape
        feat_f0_unfold = feat_f0_unfold.reshape(bs * num_coarse_match, ww, c)
        feat_f1_unfold = feat_f1_unfold.reshape(bs * num_coarse_match, ww, c)
        feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)
        feat_f0_unfold = feat_f0_unfold.reshape(bs, num_coarse_match, ww, c)
        feat_f1_unfold = feat_f1_unfold.reshape(bs, num_coarse_match, ww, c)
        return feat_f0_unfold, feat_f1_unfold

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
