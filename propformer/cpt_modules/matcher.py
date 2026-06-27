import torch
import torch.nn.functional as F
# from scipy.optimize import linear_sum_assignment
from torch import nn
import scipy.optimize
linear_sum_assignment = scipy.optimize.linear_sum_assignment

class Matcher(nn.Module):
    def __init__(self, hypercfg, asm_type='bipart'):
        super().__init__()
        self.hypercfg = hypercfg
        self.asm_type = asm_type
        self.asm_layer = 'all'  # [last, all]
        return

    @torch.no_grad()
    def forward(self, pred_pose_list, gt_pose):
        if self.asm_layer == 'last':
            raise NotImplementedError
        elif self.asm_layer == 'all':
            pred_point_ml = torch.stack([d['point'] for d in pred_pose_list], dim=1)
            pred_visible_ml = torch.stack([d['visible'] for d in pred_pose_list], dim=1)
        else:
            raise NotImplementedError

        B, G, K, _ = gt_pose['point'].size()
        _, L, M, _ = pred_point_ml.size()
        # gt_pose['cid'].flatten().reshape(B, G)
        pred_point_bgl = pred_point_ml.reshape(B, G, L, M, 2)
        pred_visible_bgl = pred_visible_ml.reshape(B, G, L, M)
        meta_indices = []
        for b in range(B):
            gt_covisible = gt_pose['visible'][b].max(0)[0]
            gt_point = gt_pose['point'][b][:, gt_covisible]
            gt_visible = gt_pose['visible'][b][:, gt_covisible].float()

            # pred_point_bgl[b] : [G,L,M,2] =>  [G,L,M,1,2]
            # gt_point          : [G,K,2]   =>  [G,1,1,K,2]
            point_cost_g = F.l1_loss(pred_point_bgl[b].unsqueeze(-2),
                                     gt_point.unsqueeze(1).unsqueeze(1),
                                     reduction='none').sum(-1).sum(1)
            point_cost = (point_cost_g * gt_visible.unsqueeze(1)).mean(0)

            # -ylogp-(1-y)log(1-p)
            # pred_visible_bgl[b] : [G,L,M] =>  [G,L,M,1]
            # gt_visible          : [G,K]   =>  [G,1,1,K]
            p = pred_visible_bgl[b].unsqueeze(-1)
            y = gt_visible.unsqueeze(1).unsqueeze(1)
            visible_cost_pos = (-y * p.log()).sum(1).mean(0)
            visible_cost_neg = (- (1 - y) * (1 - p).log()).sum(1).mean(0)
            # cost_matrix is in MxK.
            total_cost = self.hypercfg.l1 * point_cost + \
                         self.hypercfg.vp * visible_cost_pos + \
                         self.hypercfg.vn * visible_cost_neg

            meta_idx, kp_idx = linear_sum_assignment(total_cost.cpu())
            reorder = kp_idx.argsort()
            meta_ridx = meta_idx[reorder]
            kp_ridx = kp_idx[reorder]
            meta_indices.append(meta_ridx)
        return [torch.as_tensor(i, dtype=torch.int64) for i in meta_indices]
