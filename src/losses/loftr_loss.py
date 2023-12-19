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
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.
        pos_mask = pos_mask.astype(ms.int32)
        neg_mask = neg_mask.astype(ms.int32)
        if self.coarse_type == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            conf = ops.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = - ops.log(conf[pos_mask])
            loss_neg = - ops.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.coarse_type == 'focal':  # usually focal
            conf = ops.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.focal_alpha
            gamma = self.focal_gamma

            if self.sparse_spvs:  # False
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                    if self.match_type == 'sinkhorn' \
                    else conf[pos_mask]
                loss_pos = - alpha * ops.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = ops.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * ops.pow(1 - neg_conf, gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = ops.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]

                loss = c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                    if self.match_type == 'sinkhorn' \
                    else c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * ops.pow(1 - conf, gamma) * (conf).log()
                loss_neg = - alpha * ops.pow(conf, gamma) * (1 - conf).log()  ## problem
                # if weight is not None:
                loss_pos = loss_pos * weight * pos_mask
                loss_neg = loss_neg * weight * neg_mask
                return c_pos_w * (loss_pos.sum() / pos_mask.sum()) + c_neg_w * (loss_neg.sum() / neg_mask.sum())
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
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
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                correct_mask[0] = True
            else:
                return None
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
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                # sometimes there is not coarse-level gt at all.
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

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
        if self.sparse_spvs and self.match_type == 'sinkhorn':
            raise NotImplementedError()
        match_weight1 = match_masks.unsqueeze(2).tile((1, 1, c_weight.shape[2])).astype(ms.int32)
        match_weight2 = match_masks.unsqueeze(0).tile((1, c_weight.shape[1], 1)).astype(ms.int32)
        c_weight = c_weight * match_weight1 * match_weight2
        loss_c = self.compute_coarse_loss(
            conf_matrix,
            conf_matrix_gt,
            weight=c_weight)
        loss = loss_c * self.coarse_weight

        # 2. fine-level loss
        loss_f = self.compute_fine_loss(expec_f[0], expec_f_gt, match_masks)
        if loss_f is not None:
            loss += loss_f * self.fine_weight
        else:
            assert self.training is False

        return loss
