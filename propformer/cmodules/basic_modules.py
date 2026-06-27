import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from backups.mpformer.cmodules.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from mmpose.models.utils.ops import resize


def get_ref_feat(values_flatten, ref_points, spatial_shapes):
    n_heads, n_points = 1, 1
    N, Len_q, _ = ref_points.shape
    N, Len_in, d_model = values_flatten.shape
    n_levels = spatial_shapes.size(0)

    # N, Len_q, n_heads, n_levels, n_points, 2
    # A = A.view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
    # A = F.softmax(A, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
    sampling_locations = ref_points.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).expand(
        -1, -1, n_heads, n_levels, n_points, -1).contiguous()
    attention_weights = (torch.ones(N, Len_q, n_heads, n_levels, n_points).to(values_flatten) /
                         (n_levels * n_points)).contiguous()
    values_split = values_flatten.view(N, Len_in, n_heads, d_model // n_heads)
    # feat_from = MSDeformAttnFunction.apply(values_split, spatial_shapes, level_start_index,
    #                                        sampling_locations, attention_weights, 128)
    feat_from = ms_deform_attn_core_pytorch(values_split, spatial_shapes, sampling_locations, attention_weights)
    return feat_from


def get_mean_of_deep_support(feature_s, target_s):
    query_embed_list = []
    for feat_pyramid, target in zip(feature_s, target_s):
        resized_feature = resize(input=feat_pyramid[-1], size=target.shape[-2:],
                                 mode='bilinear', align_corners=False)
        target = target / (target.sum(dim=-1).sum(dim=-1)[:, :, None, None] + 1e-8)
        query_embed = target.flatten(2) @ resized_feature.flatten(2).permute(0, 2, 1)
        query_embed_list.append(query_embed)
    query_embed = torch.mean(torch.stack(query_embed_list, dim=0), 0)
    query_embed = query_embed
    return query_embed


def get_query_mask_and_order(mask_s, pe_layer):
    query_order = pe_layer(mask_s.new_zeros((mask_s.shape[0], 1, mask_s.shape[1])).to(torch.bool))
    # [bs, num_query], True indicating this query matched no actual joints.
    query_mask = (~mask_s.to(torch.bool)).squeeze(-1)
    # allow at least one valid att for these all occluded objects
    tgt_key_padding_mask_remove_all_true = query_mask.clone().to(query_mask.device)
    tgt_key_padding_mask_remove_all_true[query_mask.logical_not().sum(dim=-1) == 0, 0] = False
    query_mask = tgt_key_padding_mask_remove_all_true
    return query_mask, query_order.flatten(2).permute(0, 2, 1)


def make_pyramid_projs(backbone_dims, embed_dims, num_feature_levels):
    input_proj_list = []
    for in_channels in backbone_dims[::-1][:num_feature_levels]:
        input_proj_list.append(nn.Sequential(
            nn.Conv2d(in_channels, embed_dims, kernel_size=1),
        ))
    projs = nn.ModuleList(input_proj_list)
    for proj in projs:
        nn.init.xavier_uniform_(proj[0].weight, gain=1)
        nn.init.constant_(proj[0].bias, 0)
    return projs


def proj_pyramid(feat_pyramid, projs):
    srcs = []
    for idx in range(len(projs)):
        x = feat_pyramid[-idx - 1]
        src = projs[idx](x)
        srcs.append(src)
    return srcs


class TokenDecodeMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels=2,
                 num_layers=3):
        super(TokenDecodeMLP, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_channels, hidden_channels))
                layers.append(nn.GELU())
            else:
                layers.append(nn.Linear(hidden_channels, hidden_channels))
                layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_channels, out_channels))
        # TODO: what about tanh / 2 + center ?
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def inv_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
