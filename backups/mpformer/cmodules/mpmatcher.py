import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np


def l1_cost_matrix(predpoints, gtpoints):
    # predpoints: [L,M,2]
    # gtpoints: [K,2]

    l1_loss = F.l1_loss(predpoints.unsqueeze(2),
                        gtpoints.unsqueeze(0).unsqueeze(0),
                        reduction='none')
    return l1_loss.sum(-1)


class MPMatcher(nn.Module):
    def __init__(self, hypercfg, asm_type):
        super().__init__()
        self.hypercfg = hypercfg
        self.asm_type = asm_type
        return

    @torch.no_grad()
    def forward(self, meta_point_s, visible_s, gtdict_s):
        bs = gtdict_s['covisibles'].size(0)
        meta_indices = []
        for b in range(bs):
            covisible = gtdict_s['covisibles'][b, :, 0].bool()

            if self.asm_type == 'fixed':
                meta_ridx = np.array([i for i in range(covisible.sum().item())])
            elif self.asm_type == 'bipart':
                gt_points = gtdict_s['gtpoints'][b][covisible]

                coordinate_ml_cost = l1_cost_matrix(meta_point_s[b], gt_points)
                # use avg to collect the losss of all layers
                coordinate_cost = coordinate_ml_cost.mean(0)

                visible_cost = -torch.softmax(visible_s[b], -1)[:, -1:]

                # cost_matrix is in NxK.
                total_cost = self.hypercfg.l1 * coordinate_cost + \
                             self.hypercfg.vp * visible_cost
                meta_idx, kp_idx = linear_sum_assignment(total_cost.cpu())
                reorder = kp_idx.argsort()
                meta_ridx = meta_idx[reorder]
                kp_ridx = kp_idx[reorder]
            else:
                raise NotImplementedError

            meta_indices.append(meta_ridx)

        return [torch.as_tensor(i, dtype=torch.int64) for i in meta_indices]
