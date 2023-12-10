import mindspore as ms
from mindspore import nn, ops
from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


class LoFTR(nn.Cell):
    def __init__(self, config):
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

    def construct(self, img0, img1, mask_c0, mask_c1):
        """ 
        forward pass
        Args:
            image0: (ms.Tensor): (bs, 1, H, W)
            image1: (ms.Tensor): (bs, 1, H, W)
            mask_c0: (ms.Tensor[bool]): (bs, H, W) False indicates a padded position
            mask_c1: (ms.Tensor): (bs, H, W)

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
        match_ids, match_masks, match_conf, match_kpts_c0, match_kpts_c1 = self.coarse_matching(feat_c0, feat_c1,
                                                        hw_c0=hw_c0, hw_c1=hw_c1,
                                                        hw_i0=hw_i0, hw_i1=hw_i1,
                                                        mask_c0=mask_c0_flat,
                                                        mask_c1=mask_c1_flat)

        # Step4: crop small patch of fine-feature-map centered at coarse feature map points
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, hw_c0, hw_f0, match_ids)

        # Step4: fine-level self- and cross- attention
        feat_f0_unfold, feat_f1_unfold = self.loftr_fine_with_reshape(feat_f0_unfold, feat_f1_unfold)

        # Step5: match fine-level
        match_kpts_f0, match_kpts_f1, normed_coord_std_f = self.fine_matching(feat_f0_unfold, feat_f1_unfold,
                                                                              match_kpts_c0, match_kpts_c1,
                                                                              hw_i0, hw_f0)

        if self.training:
            pass  # TODO check loss
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
