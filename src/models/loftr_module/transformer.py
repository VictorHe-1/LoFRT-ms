import copy
from mindspore import nn, ops
import mindspore.common.initializer as init
from .linear_attention import LinearAttention, FullAttention


class LoFTREncoderLayer(nn.Cell):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Dense(d_model, d_model, has_bias=False)
        self.k_proj = nn.Dense(d_model, d_model, has_bias=False)
        self.v_proj = nn.Dense(d_model, d_model, has_bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Dense(d_model, d_model, has_bias=False)

        # feed-forward network
        self.mlp = nn.SequentialCell(
            nn.Dense(d_model*2, d_model*2, has_bias=False),
            nn.ReLU(),
            nn.Dense(d_model*2, d_model, has_bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])

    def construct(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [bs, h0w0, c]
            source (torch.Tensor): [bs, h1w1, c]
            x_mask (torch.Tensor): [bs, h0w0] (optional)
            source_mask (torch.Tensor): [bs, h1w1] (optional)
        """
        bs = x.shape[0]
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [bs, l, num_head, head_dim] l=h0w0
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [bs, s, num_head, head_dim] s=h1w1
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)  # [bs, s, num_head, head_dim]
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [bs, l, num_head, head_dim]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [bs, l, c]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(ops.cat([x, message], axis=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Cell):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.CellList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.get_parameters():
            if p.dim() > 1:
                p.set_data(init.initializer(init.XavierUniform(), p.shape, p.dtype))

    def construct(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.shape[2], "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1
