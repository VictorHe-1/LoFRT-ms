"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import mindspore as ms
from mindspore import nn, ops


def elu_feature_map(x):
    return ops.elu(x) + 1


class LinearAttention(nn.Cell):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.activate = elu_feature_map
        self.eps = eps

    def construct(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [bs, l, num_head, head_dim]
            keys: [bs, s, num_head, head_dim]
            values: [bs, s, num_head, head_dim]
            q_mask: [bs, l], 0 for pad area
            kv_mask: [bs, s]
        Returns:
            queried_values: (bs, l, num_head, head_dim)
        """
        Q = self.activate(queries)
        K = self.activate(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None].astype(Q.dtype)
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None].astype(Q.dtype)
            values = values * kv_mask[:, :, None, None].astype(Q.dtype)

        v_length = values.shape[1]
        values = values / v_length  # scale up to prevent fp16 overflow
        # KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        # (bs, s, num_head, head_dim, 1) * (bs, s, num_head, 1, head_dim) -> (bs, s, num_head, head_dim, head_dim)
        # sum 1 -> (bs, num_head, head_dim, head_dim)
        KV = (K.expand_dims(-1) * values.expand_dims(3)).sum(1)

        # (bs, l, num_head, head_dim) * (bs, 1, num_head, head_dim) -> (bs, l, num_head, head_dim)
        # sum 3 -> (bs, l, num_head)

        # if q_mask is not None:  # to prevent overflow when float16
        #     recip_z = (Q * K.sum(1).expand_dims(1)).sum(3) + ops.logical_not(q_mask[:, :, None]).astype(Q.dtype)
        #     Z = ops.reciprocal(recip_z)
        #     Z = Z - ops.logical_not(q_mask[:, :, None]).astype(Q.dtype)
        # else:
        #     recip_z = (Q * K.sum(1).expand_dims(1)).sum(3)
        #     Z = ops.reciprocal(recip_z)

        if q_mask is not None:  # to prevent 0-division
            Z = ops.where(q_mask[:, :, None], ops.reciprocal((Q * K.sum(1).expand_dims(1)).sum(3)+ self.eps), q_mask[:, :, None].astype(Q.dtype))
        else:
            Z = ops.reciprocal((Q * K.sum(1).expand_dims(1)).sum(3) + self.eps)

        # (bs, l, num_head, head_dim, 1) * (bs, 1, num_head, head_dim, head_dim) * (bs, l, num_head, 1, 1)
        # -> (bs, l, num_head, head_dim, head_dim) sum3 -> (bs, l, num_head, head_dim)
        queried_values = (Q.expand_dims(-1) * KV.expand_dims(1) * Z.expand_dims(-1).expand_dims(-1)).sum(3)
        queried_values = queried_values * v_length  # unscale
        return queried_values


class FullAttention(nn.Cell):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = ops.softmax(softmax_temp * QK, axis=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values
