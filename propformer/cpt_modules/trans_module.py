import torch.nn as nn
import torch.nn.functional as F
from propformer.cmodules.basic_modules import _get_activation_fn
from .att_module import MyDeformAttn
from mmcv.cnn import xavier_init


class DeformAttLayer(nn.Module):
    def __init__(self, num_feature_levels=3, use_self_att=False,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8, n_points=4, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.use_self_att = use_self_att
        if self.use_self_att:
            self.self_attn = nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
            self.self_dropout1 = nn.Dropout(dropout)
            self.self_norm1 = nn.LayerNorm(d_model)
            self.self_linear1 = nn.Linear(d_model, d_ffn)
            self.self_act = _get_activation_fn(activation)
            self.self_dropout2 = nn.Dropout(dropout)
            self.self_linear2 = nn.Linear(d_ffn, d_model)
            self.self_dropout3 = nn.Dropout(dropout)
            self.self_norm2 = nn.LayerNorm(d_model)

        # decoder layer
        self.deform_attn = MyDeformAttn(d_model=d_model, n_levels=num_feature_levels,
                                        n_heads=n_heads, n_points=n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        return

    def _reset_parameters(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')

        self.deform_attn._reset_parameters()
        if self.use_self_att:
            self.self_attn._reset_parameters()
        return

    def forward(self, query_embed, query_mask, query_order, values,
                ref_point, ref_pos, ref_feat, shapes, level_start):

        tgt = query_embed
        tgt = self.forward_self_att(tgt + ref_feat, query_order + ref_pos, query_mask)
        tgt2 = self.deform_attn(tgt + query_order + ref_pos + ref_feat,
                                ref_point.unsqueeze(-2).expand(-1, -1, shapes.size(0), -1).contiguous(),
                                values, shapes, level_start)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt = tgt + self.dropout3(self.linear2(self.dropout2(self.act(self.linear1(tgt)))))
        tgt = self.norm2(tgt)
        return tgt

    def forward_self_att(self, query_embed, query_pos, query_mask):
        if not self.use_self_att:
            return query_embed

        src = query_embed
        q = k = query_embed + query_pos
        src2 = self.self_attn(q, k, value=src, key_padding_mask=query_mask)[0]

        src = src + self.self_dropout1(src2)
        src = self.self_norm1(src)
        src2 = self.self_linear2(self.self_dropout2(self.self_act(self.self_linear1(src))))
        src = src + self.self_dropout3(src2)
        src = self.self_norm2(src)
        return src


class NeighborLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8, ltype='ex1'):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.self_dropout1 = nn.Dropout(dropout)
        self.self_norm1 = nn.LayerNorm(d_model)
        self.self_linear1 = nn.Linear(d_model, d_ffn)
        self.self_act = _get_activation_fn(activation)
        self.self_dropout2 = nn.Dropout(dropout)
        self.self_linear2 = nn.Linear(d_ffn, d_model)
        self.self_dropout3 = nn.Dropout(dropout)
        self.self_norm2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.ltype = ltype

    def _reset_parameters(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self.self_attn._reset_parameters()

    def forward(self, query_embed, query_pos, link):

        if self.ltype == 'ex1':
            link_ex = link[:, None].expand(-1, self.n_heads, -1, -1).flatten(0, 1)
        elif self.ltype == 'ex2':
            link_ex = link[None].expand(self.n_heads, -1, -1, -1).flatten(0, 1)
        elif self.ltype == 'zero':
            link_ex = link[None].expand(self.n_heads, -1, -1, -1).flatten(0, 1) * 0
        else:
            raise NotImplementedError(f'{self.ltype}')

        src = query_embed
        q = k = query_embed + query_pos
        src2 = self.self_attn(q, k, value=src, attn_mask=link_ex)[0]
        src = src + self.self_dropout1(src2)
        src = self.self_norm1(src)
        src2 = self.self_linear2(self.self_dropout2(self.self_act(self.self_linear1(src))))
        src = src + self.self_dropout3(src2)
        src = self.self_norm2(src)
        return src
