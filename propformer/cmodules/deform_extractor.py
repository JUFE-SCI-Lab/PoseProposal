import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from capeformer.models.keypoint_heads.two_stage_head import inverse_sigmoid, TokenDecodeMLP
from mmcv.cnn.bricks.transformer import build_positional_encoding
from backups.mpformer.cmodules.ops.modules import MSDeformAttn
from backups.mpformer.cmodules.ops.functions import MSDeformAttnFunction
from backups.mpformer.cmodules.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch

from typing import Optional, List
from torch import Tensor
from .basic_modules import MLP, _get_activation_fn


class DeformableExtractor(nn.Module):
    def __init__(self, num_feature_levels=3, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8, n_points=4, **kwargs):
        super().__init__()
        self.out_layer_num = kwargs['out_layer_num']
        self.use_self_att = kwargs['use_self_att']

        self.dec_layers = nn.ModuleList()
        for l in range(self.out_layer_num):
            self.dec_layers.append(DecoderLayer(n_levels=num_feature_levels, n_heads=n_heads, d_ffn=d_ffn,
                                                n_points=n_points, activation=activation, d_model=d_model,
                                                ref_feat_aware=kwargs['ref_feat_aware']))

        if self.use_self_att:
            self.self_att_layers = nn.ModuleList()
            self.self_att_norms = nn.ModuleList()
            for _ in range(self.out_layer_num):
                self.self_att_layers.append(nn.MultiheadAttention(self.global_dim, num_heads=n_points, dropout=dropout))
                self.self_att_norms.append(nn.LayerNorm(self.global_dim))
        self.pos_embed_proj = MLP(d_model, d_model, d_model, 2)

        self.out_norm = kwargs['out_norm']
        self.return_intermediate = kwargs['return_intermediate']
        return

    def forward(self, query_embed, memory_pyramid, tgt_key_padding_mask, pos,
                pe_layer=None, initial_proposals=None, kpt_branch=None):

        value_feat_flatten = []
        spatial_shapes = []
        for l, featmap in enumerate(memory_pyramid):
            bs, c, h, w = featmap.shape
            spatial_shapes.append((h, w))
            featmap = featmap.flatten(2).transpose(1, 2)
            value_feat_flatten.append(featmap)

        value_feat_flatten = torch.cat(value_feat_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=query_embed.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        latest_points = initial_proposals
        latest_points_unsig = inverse_sigmoid(initial_proposals)
        query_points = []
        intermediate = []

        tgt_key_padding_mask_remove_all_true = tgt_key_padding_mask.clone().to(tgt_key_padding_mask.device)
        tgt_key_padding_mask_remove_all_true[tgt_key_padding_mask.logical_not().sum(dim=-1) == 0, 0] = False
        # pos_embed = torch.cat((pos_embed, support_order_embed))

        output = query_embed
        for lidx, dec_layer in enumerate(self.dec_layers):
            support_order_embed = pos[-output.size(0):]
            query_pos_embed = pe_layer.forward_coordinates(latest_points)
            query_pos_embed = query_pos_embed.transpose(0, 1)
            query_pos_embed = self.pos_embed_proj(query_pos_embed)

            output = self.forward_self_att(output, query_pos_embed, lidx, tgt_key_padding_mask_remove_all_true)

            output = dec_layer(output, query_pos_embed, support_order_embed,
                               value_feat_flatten, latest_points,
                               spatial_shapes, level_start_index)
            if self.return_intermediate:
                intermediate.append(self.out_norm(output))

            new_query_point_unsig = latest_points_unsig + kpt_branch[lidx](output.transpose(0, 1))
            new_query_point = new_query_point_unsig.sigmoid()

            latest_points_unsig = new_query_point_unsig.detach()
            latest_points = new_query_point.detach()
            query_points.append(new_query_point)

        if self.out_norm is not None:
            output = self.out_norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), query_points
        else:
            return output.unsqueeze(0), query_points

    def forward_self_att(self, query_embed, query_pos, layer_idx, tgt_key_padding_mask_remove_all_true):
        if not self.use_self_att:
            return query_embed

        q = k = self.with_pos_embed(query_embed, query_pos).transpose(0, 1)
        # [L,B,C]
        tgt2 = self.self_att_layers[layer_idx](q, k, value=query_embed.transpose(0, 1))[0]
        tgt = query_embed + F.dropout(tgt2.transpose(0, 1), 0.1)
        tgt = self.self_att_norms[layer_idx](tgt)
        return tgt


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, ref_feat_aware=True):
        super().__init__()

        self.deform_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ref_feat_aware = ref_feat_aware
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_points = n_points
        if ref_feat_aware:
            self.ref_feat_fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU()
            )

    def forward(self, query_embed, query_pos_embed, support_order_embed,
                values_flatten, reference_points,
                spatial_shapes, level_start_index):
        '''
            output, query_pos_embed, support_order_embed,
            value_feat_flatten, latest_points,
            spatial_shapes, level_start_index
        '''
        L = spatial_shapes.size(0)
        ref_points_ex = reference_points.unsqueeze(-2).expand(-1, -1, L, -1).contiguous()

        if self.ref_feat_aware:
            Len_q, N, _ = query_embed.shape
            N, Len_in, _ = values_flatten.shape
            # N, Len_q, n_heads, n_levels, n_points, 2
            # A = A.view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
            # A = F.softmax(A, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
            n_heads, n_points = 1, 1
            sampling_locations = ref_points_ex.unsqueeze(2).unsqueeze(-2).expand(
                -1, -1, n_heads, -1, -1, -1).contiguous()
            attention_weights = (torch.ones(N, Len_q, n_heads, self.n_levels, n_points).to(
                reference_points) / (self.n_levels * n_points)).contiguous()
            values_split = values_flatten.view(N, Len_in, n_heads, self.d_model // n_heads)
            # feat_from = MSDeformAttnFunction.apply(values_split, spatial_shapes, level_start_index,
            #                                        sampling_locations, attention_weights, 128)

            feat_from = ms_deform_attn_core_pytorch(values_split, spatial_shapes, sampling_locations, attention_weights)
            query_from = (query_embed + query_pos_embed + support_order_embed).transpose(0, 1).contiguous()
            atten_from = self.ref_feat_fusion(torch.cat([feat_from.contiguous(), query_from], -1)).contiguous()
        else:
            atten_from = (query_embed + query_pos_embed + support_order_embed).transpose(0, 1)

        query_embed2 = self.deform_attn(atten_from,
                                        ref_points_ex, values_flatten,
                                        spatial_shapes, level_start_index)
        query_embed = query_embed.transpose(0, 1) + self.dropout1(query_embed2)
        query_embed = self.norm1(query_embed)

        query_embed = query_embed + self.dropout3(
            self.linear2(self.dropout2(self.activation(self.linear1(query_embed)))))
        query_embed = self.norm2(query_embed)

        return query_embed.transpose(0, 1)
