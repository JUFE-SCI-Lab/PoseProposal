import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from .trans_module import DeformAttLayer, NeighborLayer
from propformer.cmodules.basic_modules import get_ref_feat, MLP, inv_sigmoid, TokenDecodeMLP

eps = 1e-3


class PredictHead(nn.Module):
    def __init__(self, use_self_att=False, num_feature_levels=3,
                 d_model=256, nhead=8, npoint=4, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.kwargs = kwargs
        self.point_head = TokenDecodeMLP(in_channels=self.d_model, hidden_channels=self.d_model, out_channels=2)
        if self.kwargs['vl_type'] == 'sigmoid':
            self.visible_head = TokenDecodeMLP(in_channels=self.d_model, hidden_channels=self.d_model, out_channels=1)
            self.link_head = TokenDecodeMLP(in_channels=self.d_model, hidden_channels=self.d_model, out_channels=1)
        elif self.kwargs['vl_type'] == 'softmax':
            self.visible_head = TokenDecodeMLP(in_channels=self.d_model, hidden_channels=self.d_model, out_channels=2)
            self.link_head = TokenDecodeMLP(in_channels=self.d_model, hidden_channels=self.d_model, out_channels=2)
        else:
            raise NotImplementedError()

        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')

        self.point_deformatt_layer = DeformAttLayer(num_feature_levels=num_feature_levels,
                                                    use_self_att=use_self_att,
                                                    n_points=npoint, n_heads=nhead)
        self.point_deformatt_layer._reset_parameters()

        if self.kwargs['use_neighbors'] != 'off':
            self.neighbor_layer = NeighborLayer(n_heads=nhead, ltype=self.kwargs['use_neighbors'])
            self.neighbor_layer._reset_parameters()
        if self.kwargs['use_link_defatt']:
            self.link_deformatt_layer = DeformAttLayer(num_feature_levels=num_feature_levels,
                                                       use_self_att=use_self_att,
                                                       n_points=npoint, n_heads=nhead)
            self.link_deformatt_layer._reset_parameters()
        if self.kwargs['use_link_pos']:
            self.link_pos_proj = MLP(d_model, d_model, d_model, 2)
            for m in self.link_pos_proj.modules():
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    xavier_init(m, distribution='uniform')
        self.ref_feat_factor = kwargs['ref_feat_factor']
        return

    def forward(self, featpyramid, last_pose, pos_embed_proj, pe_layer):
        query_embed = last_pose['embed']

        last_point = last_pose['point']
        last_visible = last_pose['visible']
        last_link = last_pose['link']

        last_point_det = last_point.detach()
        last_visible_det = last_visible.detach()
        last_link_det = last_link.detach()

        B, N, D = query_embed.shape
        query_mask = None
        query_order = pe_layer(query_embed.new_zeros((B, 1, N)).to(torch.bool)).flatten(2).permute(0, 2, 1)
        shapes = torch.as_tensor([p.shape[-2:] for p in featpyramid], dtype=torch.long).to(query_embed.device)
        level_start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
        values = torch.cat([p.flatten(2).transpose(1, 2) for p in featpyramid], 1)

        ref_pos = pos_embed_proj(pe_layer.forward_coordinates(last_point_det))
        ref_feat = get_ref_feat(values, last_point_det, shapes) if self.ref_feat_factor != 0 else 0.
        query_embed = self.point_deformatt_layer(query_embed, query_mask, query_order, values,
                                                 last_point_det, ref_pos, ref_feat, shapes, level_start)
        # TODO: structure refine
        if self.kwargs['use_neighbors'] != 'off':
            query_embed = self.neighbor_layer(query_embed, ref_pos, last_link_det)
        else:
            pass

        # Predict point and visible
        pred_point = (inv_sigmoid(last_point) + self.point_head(query_embed)).sigmoid()

        if self.kwargs['vl_type'] == 'sigmoid':
            pred_visible = (inv_sigmoid(last_visible) + self.visible_head(query_embed).squeeze(-1)).sigmoid()
        elif self.kwargs['vl_type'] == 'softmax':
            pred_visible = self.visible_head(query_embed).softmax(-1)[..., -1]
        else:
            raise NotImplementedError()

        # Predict link
        pair_embed = query_embed.unsqueeze(2) + query_embed.unsqueeze(1)

        pred_point_det = pred_point.detach()
        pred_visible_det = pred_visible.detach()

        if self.kwargs['use_link_defatt']:
            pair_point = (pred_point_det.unsqueeze(2) + pred_point_det.unsqueeze(1)) / 2
            pair_embed_f = pair_embed.reshape(B, -1, D)
            pair_point_f = pair_point.reshape(B, -1, 2)
            pair_order = pe_layer(pair_embed_f.new_zeros((B, 1, N * N)).to(torch.bool)).flatten(2).permute(0, 2, 1)
            pair_ref_pos = pos_embed_proj(pe_layer.forward_coordinates(pair_point_f))
            pair_ref_feat = get_ref_feat(values, pair_point_f, shapes) if self.ref_feat_factor != 0 else 0.
            pair_embed_r = self.link_deformatt_layer(pair_embed_f, None, pair_order, values,
                                                     pair_point_f, pair_ref_pos, pair_ref_feat, shapes, level_start)
        else:
            pair_embed_r = pair_embed.reshape(B, -1, D)

        if self.kwargs['use_link_pos']:
            pred_point_pos_embed = self.link_pos_proj(pe_layer.forward_coordinates(pred_point_det))
            se_pos_embed = pred_point_pos_embed.unsqueeze(2) + pred_point_pos_embed.unsqueeze(1)
            pair_embed_r = pair_embed_r + se_pos_embed.reshape(B, -1, D)

        if self.kwargs['vl_type'] == 'sigmoid':
            pred_link = (inv_sigmoid(last_link) +
                         self.link_head(pair_embed_r).squeeze(-1).reshape(B, N, N)).sigmoid()
        elif self.kwargs['vl_type'] == 'softmax':
            pred_link = self.link_head(pair_embed_r).reshape(B, N, N, 2).softmax(-1)[..., -1]
        else:
            raise NotImplementedError()

        # Mask links with visibles
        # pred_link_masked = pred_link * (pred_visible_det.unsqueeze(2) * pred_visible_det.unsqueeze(1)).sqrt()
        pred_link_masked = pred_link * (pred_visible_det.unsqueeze(2) / 2 + pred_visible_det.unsqueeze(1) / 2)

        # if self.vizing:
        #     pred_pose = {}
        #     pred_pose['embed'] = query_embed
        #     pred_pose['point'] = pred_point
        #     pred_pose['visible'] = pred_visible
        #     pred_pose['link'] = pred_link_masked
        #     return pred_pose

        pred_pose = {}
        pred_pose['embed'] = query_embed
        pred_pose['point'] = pred_point.clamp(min=eps, max=1 - eps)
        pred_pose['visible'] = pred_visible.clamp(min=eps, max=1 - eps)
        pred_pose['link'] = pred_link_masked.clamp(min=eps, max=1 - eps)
        return pred_pose
