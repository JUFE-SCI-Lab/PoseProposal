import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
from typing import Optional, List
from .basic_modules import _get_activation_fn, MLP, _get_clones, inv_sigmoid


class MTransformerDecoder(nn.Module):
    def __init__(self, d_model, decoder_layer, num_layers, norm=None, return_intermediate=True,
                 own_kpt_branch=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.ref_point_head = MLP(d_model, d_model, d_model, 2)
        self.own_kpt_branch = own_kpt_branch

    def forward(self, tgt, memory_pyramid,
                tgt_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None,
                pe_layer=None, initial_proposals=None, kpt_branch=None):
        output = tgt
        intermediate = []
        query_coordinates = initial_proposals.detach()
        query_points = [initial_proposals.detach()]

        tgt_key_padding_mask_remove_all_true = tgt_key_padding_mask.clone().to(tgt_key_padding_mask.device)
        tgt_key_padding_mask_remove_all_true[tgt_key_padding_mask.logical_not().sum(dim=-1) == 0, 0] = False

        for lidx, layer in enumerate(self.layers):
            if lidx == 0:  # use positional embedding form inital proposals
                query_pos_embed = query_pos.transpose(0, 1)
            else:
                query_pos_embed = pe_layer.forward_coordinates(query_coordinates)
                query_pos_embed = query_pos_embed.transpose(0, 1)
            query_pos_embed = self.ref_point_head(query_pos_embed)

            output = layer(output, memory_pyramid,
                           tgt_key_padding_mask=tgt_key_padding_mask_remove_all_true,
                           pos=pos, query_pos=query_pos_embed)

            if self.return_intermediate:
                intermediate.append(self.norm(output))

            query_coordinates_unsigmoid = inv_sigmoid(query_coordinates)
            new_query_coordinates = (query_coordinates_unsigmoid + kpt_branch[lidx](output.transpose(0, 1))).sigmoid()
            query_coordinates = new_query_coordinates.detach()
            query_points.append(new_query_coordinates)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), query_points

        return output.unsqueeze(0), query_points


class MTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        self.choker = nn.Linear(in_features=2 * d_model, out_features=d_model)
        # Implementation of Feedforward model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory_pyramid,
                tgt_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        memory = memory_pyramid[0]
        q = k = self.with_pos_embed(tgt, query_pos + pos[memory.shape[0]:])
        tgt2 = self.self_attn(q, k, value=tgt, key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # concatenate the positional embedding with the content feature,
        # instead of direct addition
        cross_attn_q = torch.cat((tgt, query_pos + pos[memory.shape[0]:]), dim=-1)
        cross_attn_k = torch.cat((memory, pos[:memory.shape[0]]), dim=-1)
        tgt2 = self.multihead_attn(query=cross_attn_q, key=cross_attn_k, value=memory)[0]
        tgt = tgt + self.dropout2(self.choker(tgt2))
        tgt = self.norm2(tgt)
        tgt = tgt + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt)))))
        tgt = self.norm3(tgt)
        return tgt
