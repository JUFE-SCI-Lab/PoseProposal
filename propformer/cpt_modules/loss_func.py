import torch
import torch.nn.functional as F


def get_pose_loss(hypercfg, meta_to_kp, pred_pose_list, gt_pose):
    L = len(pred_pose_list)
    l1_loss_lists = []
    vp_loss_lists, vn_loss_lists = [], []
    mp_loss_lists, mn_loss_lists = [], []
    for _ in range(L):
        l1_loss_lists.append([])
        vp_loss_lists.append([])
        vn_loss_lists.append([])
        mp_loss_lists.append([])
        mn_loss_lists.append([])

    B, G, K, _ = gt_pose['point'].size()
    M = pred_pose_list[0]['point'].size(1)
    pred_pose_bgl = {}
    pred_pose_bgl['point'] = torch.stack([p['point'] for p in pred_pose_list], dim=1).reshape(B, G, L, M, 2)
    pred_pose_bgl['visible'] = torch.stack([p['visible'] for p in pred_pose_list], dim=1).reshape(B, G, L, M)
    pred_pose_bgl['link'] = torch.stack([p['link'] for p in pred_pose_list], dim=1).reshape(B, G, L, M, M)

    for b, m2k in enumerate(meta_to_kp):
        gt_covisible = gt_pose['visible'][b].max(0)[0]
        gt_point = gt_pose['point'][b][:, gt_covisible]
        gt_visible = gt_pose['visible'][b][:, gt_covisible].float()

        # asm_point : [G,L,K,2] =>  [G,L,K,2]
        # gt_point  : [G,K,2]   =>  [G,1,K,2]
        asm_point = pred_pose_bgl['point'][b][:, :, m2k]
        l1_loss_GL = F.l1_loss(asm_point, gt_point.unsqueeze(1), reduction='none').sum(-1)
        l1_loss_L = (l1_loss_GL * gt_visible.unsqueeze(1)).mean(0).mean(-1)

        # -ylogp-(1-y)log(1-p)
        p = pred_pose_bgl['visible'][b][:, :, m2k]
        y = gt_visible.unsqueeze(1)
        vp_loss_L = (-y * p.log()).mean(0).sum(-1) / M
        vna_loss_L = (- (1 - y) * (1 - p).log()).mean(0).sum(-1)

        unassigned = torch.ones(M).to(gt_point).bool()
        unassigned[m2k] = False
        unasm_visible_GL = pred_pose_bgl['visible'][b][:, :, unassigned]
        vnu_loss_L = (- (1 - unasm_visible_GL).log()).mean(0).sum(-1)
        vn_loss_L = (vna_loss_L + vnu_loss_L) / M

        # -ylogp-(1-y)log(1-p)
        gt_link_GL = gt_pose['link'][b][:, gt_covisible][:, :, gt_covisible].unsqueeze(1)
        pred_link_GL = pred_pose_bgl['link'][b][:, :, m2k][..., m2k]
        mp_loss_L = -(gt_link_GL * torch.log(pred_link_GL)).mean(0).sum([1, 2]) / M / M
        mn_loss_L = -((1 - gt_link_GL) * torch.log(1 - pred_link_GL)).mean(0).sum([1, 2]) / M / M

        for l in range(L):
            l1_loss_lists[l].append(l1_loss_L[l])
            vp_loss_lists[l].append(vp_loss_L[l])
            vn_loss_lists[l].append(vn_loss_L[l])
            mp_loss_lists[l].append(mp_loss_L[l])
            mn_loss_lists[l].append(mn_loss_L[l])

    loss_dict = {}
    for l in range(L):
        loss_dict[f'loss_L_{l}'] = torch.stack(l1_loss_lists[l]).nanmean() * hypercfg.l1
        loss_dict[f'loss_vp_{l}'] = torch.stack(vp_loss_lists[l]).nanmean() * hypercfg.vp
        loss_dict[f'loss_vn_{l}'] = torch.stack(vn_loss_lists[l]).nanmean() * hypercfg.vn
        loss_dict[f'loss_mp_{l}'] = torch.stack(mp_loss_lists[l]).nanmean() * hypercfg.mp
        loss_dict[f'loss_mn_{l}'] = torch.stack(mn_loss_lists[l]).nanmean() * hypercfg.mn
    return loss_dict
