import torch
import torch.nn.functional as F
from torch import nn


def get_ref_loss(hypercfg, ref_point, gtdict):
    l1_loss_lists = [[] for _ in range(30)]

    for b in range(len(ref_point)):
        covisible = gtdict['covisibles'][b, :, 0]
        gt_points = gtdict['gtpoints'][b][covisible]

        l1_loss_ml = F.l1_loss(ref_point[b, :, covisible], gt_points, reduction='none').sum(-1).mean(-1)
        for l, l1_loss in enumerate(l1_loss_ml):
            l1_loss_lists[l].append(l1_loss)

    loss_dict = {}
    for l, ref_loss_list in enumerate(l1_loss_lists):
        if ref_loss_list:
            loss_dict[f'loss_r{l}'] = torch.stack(ref_loss_list).nanmean() * hypercfg.l1 * hypercfg.ref

    return loss_dict


def get_meta_loss(hypercfg, meta_to_kp, meta_point, meta_visible, gtdict):
    l1_loss_lists = [[] for _ in range(30)]
    vp_loss_list = []
    vn_loss_list = []

    for b, m2k in enumerate(meta_to_kp):
        covisible = gtdict['covisibles'][b, :, 0]
        gt_points = gtdict['gtpoints'][b][covisible]

        meta_points_ml = meta_point[b][:, m2k]
        assert meta_points_ml.shape[1:] == gt_points.shape
        l1_loss_ml = F.l1_loss(meta_points_ml, gt_points, reduction='none').sum(-1).mean(-1)
        for l, l1_loss in enumerate(l1_loss_ml):
            l1_loss_lists[l].append(l1_loss)

        vscores_pos = meta_visible[b][m2k]
        vp_loss = F.cross_entropy(vscores_pos, torch.ones_like(vscores_pos[:, 0]).long())
        vp_loss_list.append(vp_loss)
        unassigned_idx = torch.ones_like(meta_visible[b, :, 0]).bool()
        unassigned_idx[m2k] = False
        vscores_neg = meta_visible[b][unassigned_idx]
        vn_loss = F.cross_entropy(vscores_neg, torch.zeros_like(vscores_neg[:, 0]).long())
        vn_loss_list.append(vn_loss)

    loss_dict = {}
    for l, ref_loss_list in enumerate(l1_loss_lists):
        if ref_loss_list:
            loss_dict[f'loss_m{l}'] = torch.stack(ref_loss_list).nanmean() * hypercfg.l1 * hypercfg.meta

    loss_dict[f'loss_vp'] = torch.stack(vp_loss_list).nanmean() * hypercfg.vp * hypercfg.meta
    loss_dict[f'loss_vn'] = torch.stack(vn_loss_list).nanmean() * hypercfg.vn * hypercfg.meta
    return loss_dict
