import mindspore as ms
from mindspore import nn, ops


class LoFTRLoss(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['loftr']['loss']
        self.match_type = self.config['loftr']['match_coarse']['match_type']
        self.sparse_spvs = self.config['loftr']['match_coarse']['sparse_spvs']

        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']

        self.coarse_weight = self.loss_config['coarse_weight']
        self.fine_weight = self.loss_config['fine_weight']
        self.coarse_type = self.loss_config['coarse_type']
        self.focal_alpha = self.loss_config['focal_alpha']
        self.focal_gamma = self.loss_config['focal_gamma']

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (ms.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (ms.Tensor): (N, HW0, HW1)
            weight (ms.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        pos_mask = pos_mask.astype(ms.int32)
        neg_mask = neg_mask.astype(ms.int32)

        if self.coarse_type == 'focal':
            conf = ops.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.focal_alpha
            gamma = self.focal_gamma
            loss_pos = - alpha * ops.pow(1 - conf, gamma) * (conf).log()
            loss_neg = - alpha * ops.pow(conf, gamma) * (1 - conf).log()
            loss_pos = loss_pos * weight * pos_mask
            loss_neg = loss_neg * weight * neg_mask
            return c_pos_w * (loss_pos.sum() / pos_mask.sum()) + c_neg_w * (loss_neg.sum() / neg_mask.sum())
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))

    def compute_fine_loss(self, expec_f, expec_f_gt, match_masks):
        if self.fine_type == 'l2_with_std':  # usually l2_with_std
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt, match_masks)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (ms.Tensor): [M, 2] <x, y>
            expec_f_gt (ms.Tensor): [M, 2] <x, y>
        """
        correct_mask = ops.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt, match_masks):
        """
        Args:
            expec_f (ms.Tensor): [M, 3] <x, y, std>
            expec_f_gt (ms.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = ops.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / ops.clamp(std, min=1e-10)
        weight = (inverse_std / ops.mean(inverse_std))  # avoid minizing loss through increase std
        weight = ops.stop_gradient(weight)
        # corner case: no correct coarse match found
        correct_mask = correct_mask.astype(ms.int32) * match_masks[0].astype(ms.int32)
        correct_mask = correct_mask.bool()

        # l2 loss with std
        correct_mask = correct_mask.astype(ms.int32)
        offset_l2 = ((expec_f_gt - expec_f[:, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight * correct_mask).sum() / correct_mask.sum()

        return loss

    def compute_c_weight(self, mask0, mask1):
        """ compute element-wise weights for computing coarse-level loss. """
        mask0 = mask0.astype(ms.int32)
        mask1 = mask1.astype(ms.int32)
        if mask0 is not None:
            c_weight = (mask0.flatten(start_dim=-2)[..., None] * mask1.flatten(start_dim=-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def construct(self,
                  expec_f,
                  expec_f_gt,
                  mask0,
                  mask1,
                  conf_matrix,
                  conf_matrix_gt,
                  match_masks
                  ):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(mask0, mask1)
        c_weight = ops.stop_gradient(c_weight)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            conf_matrix,
            conf_matrix_gt,
            weight=c_weight)
        loss = loss_c * self.coarse_weight

        # 2. fine-level loss
        loss_f = self.compute_fine_loss(expec_f[0], expec_f_gt, match_masks)
        loss += loss_f * self.fine_weight

        return loss
