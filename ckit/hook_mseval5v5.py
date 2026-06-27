# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings

import torch
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm
import os.path as osp
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import torch
import numpy as np
# from vkits.vpose import *
import time

MMPOSE_GREATER_KEYS = [
    'acc', 'ap', 'ar', 'pck', 'auc', '3dpck', 'p-3dpck', '3dauc', 'p-3dauc',
    'pcp', 'pose_mAP'
]
MMPOSE_LESS_KEYS = ['loss', 'epe', 'nme', 'mpjpe', 'p-mpjpe', 'n-mpjpe']


class MSEvalHook(_EvalHook):

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 test_fn=None,
                 greater_keys=MMPOSE_GREATER_KEYS,
                 less_keys=MMPOSE_LESS_KEYS,
                 **eval_kwargs):

        if test_fn is None:
            from mmpose.apis import single_gpu_test
            test_fn = single_gpu_test

        # to be compatible with the config before v0.16.0

        # remove "gpu_collect" from eval_kwargs
        if 'gpu_collect' in eval_kwargs:
            warnings.warn(
                '"gpu_collect" will be deprecated in EvalHook.'
                'Please remove it from the config.', DeprecationWarning)
            _ = eval_kwargs.pop('gpu_collect')

        # update "save_best" according to "key_indicator" and remove the
        # latter from eval_kwargs
        if 'key_indicator' in eval_kwargs or isinstance(save_best, bool):
            warnings.warn(
                '"key_indicator" will be deprecated in EvalHook.'
                'Please use "save_best" to specify the metric key,'
                'e.g., save_best="AP".', DeprecationWarning)

            key_indicator = eval_kwargs.pop('key_indicator', 'AP')
            if save_best is True and key_indicator is None:
                raise ValueError('key_indicator should not be None, when '
                                 'save_best is set to True.')
            save_best = key_indicator

        super().__init__(dataloader, start, interval, by_epoch, save_best,
                         rule, test_fn, greater_keys, less_keys, **eval_kwargs)

        self.metrics_history = []
        self.pths = [0.05, 0.1, 0.15, 0.2]
        self.sths = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        return

    # def get_metric_given_valid_th(self, results, point_th, score_th, epoch=None):
    #     # Global Asignment with the Last Prediction
    #     cost_globals = {}
    #     for pred_pose_list, gt_pose, cids in results:
    #         pred_point = pred_pose_list[-1]['point'][0]
    #         pred_visible = pred_pose_list[-1]['visible'][0]
    #         gt_point = gt_pose['point'][0]
    #         gt_visible = gt_pose['visible'][0]
    #         cost_v = (pred_visible.unsqueeze(1) >= score_th) & gt_visible.unsqueeze(0)
    #         cost_l2 = (pred_point.unsqueeze(1) - gt_point.unsqueeze(0)).square().sum(-1).sqrt() <= point_th
    #         cost_per = 1 - (cost_v & cost_l2).float()
    #         cid = cids[0].item()
    #         if cid in cost_globals:
    #             cost_globals[cid] = cost_globals[cid] + cost_per
    #         else:
    #             cost_globals[cid] = cost_per
    #
    #     # Solve the assignment for each class
    #     cid_2_m2k = {}
    #     for cid, cost_global in cost_globals.items():
    #         meta_idx, kp_idx = linear_sum_assignment(cost_global.cpu())
    #         reorder = kp_idx.argsort()
    #         meta_ridx = meta_idx[reorder]
    #         kp_ridx = kp_idx[reorder]
    #         cid_2_m2k[cid] = meta_ridx
    #
    #     # Compute PR,RR for each class
    #     point_PRs, point_RRs, link_PRs, link_RRs = {}, {}, {}, {}
    #     for cid in cid_2_m2k.keys():
    #         point_PRs[cid] = []
    #         point_RRs[cid] = []
    #         link_PRs[cid] = []
    #         link_RRs[cid] = []
    #     for r, (pred_pose_list, gt_pose, cids) in enumerate(results):
    #         pred_point = pred_pose_list[-1]['point'][0]
    #         pred_visible = pred_pose_list[-1]['visible'][0]
    #         # pred_link = pred_pose_list[-1]['link'][0]
    #         gt_point = gt_pose['point'][0]
    #         gt_visible = gt_pose['visible'][0]
    #         # gt_link = gt_pose['link'][0]
    #         cid = cids[0].item()
    #         m2k = cid_2_m2k[cid]
    #         l2_condition = (pred_point[m2k][gt_visible] - gt_point[gt_visible]).square().sum(-1).sqrt() <= point_th
    #         vis_condition = pred_visible[m2k][gt_visible] >= score_th
    #
    #         pred_visible_num = (pred_visible > score_th).sum()
    #         gt_visible_num = gt_visible.sum()
    #
    #         point_hit = (l2_condition & vis_condition).sum()
    #         point_PR = point_hit / pred_visible_num if pred_visible_num != 0 else (pred_visible_num + 1)
    #         point_RR = point_hit / gt_visible_num
    #         point_PRs[cid].append(point_PR)
    #         point_RRs[cid].append(point_RR)
    #
    #
    #         '''
    #         gt_m = gt_link[gt_visible][:, gt_visible]
    #         target_link = torch.zeros_like(pred_link).bool()
    #         m2kv = m2k[gt_visible.cpu()]
    #         for x, y in zip(*torch.where(gt_m)):
    #             target_link[m2kv[x.item()], m2kv[y.item()]] = True
    #         assert (target_link[m2kv][:, m2kv] == gt_m).min().item()
    #         pred_link_dual = (pred_link + pred_link.T) / 2
    #
    #         gt_link_num = target_link.triu(1).sum()
    #         pred_link_num = (pred_link_dual > score_th).triu(1).sum()
    #         link_hit = (pred_link_dual[target_link.triu(1)] >= score_th).sum()
    #         link_PR = link_hit / pred_link_num if pred_link_num != 0 else (pred_link_num + 1)
    #         link_RR = link_hit / gt_link_num
    #         link_PRs[cid].append(link_PR)
    #         link_RRs[cid].append(link_RR)
    #         if gt_link_num == 0 or link_hit == 0:
    #             d = 1
    #         '''
    #         '''
    #         if ((r + 1) % self.eval_kwargs['viz_interval'] == 0) and epoch is not None:
    #             dir_path = self.eval_kwargs['res_folder'] + 'eval_viz'
    #             os.makedirs(dir_path, exist_ok=True)
    #             iid = iids[0].item()
    #             data = self.dataloader.dataset.__getitem__(iid)
    #             mean = torch.tensor([0.485, 0.456, 0.406])[None, None, :]
    #             std = torch.tensor([0.229, 0.224, 0.225])[None, None, :]
    #             img_n = (data['img_q'].permute(1, 2, 0) * std + mean).numpy() * 255
    #             row1 = {}
    #             row1[f'Pred'] = draw_pred_pose(img=img_n, points=pred_point,
    #                                            visibles=pred_visible, links=pred_link)
    #             row1[f'Assign'] = draw_pred_pose(img=img_n, points=pred_point[m2kv],
    #                                              visibles=pred_visible[m2kv], links=pred_link[m2kv][:, m2kv])
    #             imged_gt = draw_gt_pose(img=img_n, points=data['point_q'],
    #                                     visibles=data['visible_q'], links=data['link_q'])
    #             pstr = f'p{point_PR.item():.1%}; {point_RR.item():.1%}'
    #             lstr = f'l{link_PR.item():.1%}; {link_RR.item():.1%}'
    #             row0 = {pstr: img_n / 255, lstr: imged_gt}
    #             viz_dict_list([row0, row1], fpath=f'{dir_path}/{epoch}_{iid}.jpg')
    #         '''
    #
    #     return point_PRs, point_RRs#, link_PRs, link_RRs

    def get_metric_given_valid_th(self, results, point_th, score_th, epoch=None):
        # # 1. 按 90/10 切分：每 100 个里前 90 做匹配，后 10 算指标
        # assign_set, eval_set = [], []
        # for i in range(0, len(results), 100):
        #     block = results[i:i+100]
        #     assign_set.extend(block[:50])   # 90 % 建匹配
        #     eval_set.extend(block[50:])     # 10 % 算 PR/RR

        # 1. 按类别分组
        results_by_cid = {}
        for result in results:
            # 根据你的results结构获取cid
            # 假设results每个元素是 (pred_pose_list, gt_pose, cids, ...)
            pred_pose_list, gt_pose, cids = result[0], result[1], result[2]
            cid = cids[0].item() if hasattr(cids, '__getitem__') else cids.item()

            if cid not in results_by_cid:
                results_by_cid[cid] = []
            results_by_cid[cid].append((pred_pose_list, gt_pose, cids))
        # 2. 对每个类别按50/50划分
        assign_set, eval_set = [], []
        for cid, cid_results in results_by_cid.items():
            total_samples = len(cid_results)
            split_point = total_samples // 2  # 50%分界点

            # 前50%用于匹配
            assign_set.extend(cid_results[:split_point])
            # 后50%用于评估
            eval_set.extend(cid_results[split_point:])

        # # 初始化两个列表分别存储前60个和后40个元素
        # assign_set = []  # 存储每100个中的前60个
        # eval_set = []  # 存储每100个中的后40个

        # # 遍历results列表，步长为100
        # for i in range(0, len(results), 100):
        #     # 获取当前100个元素的块
        #     block = results[i:i + 100]
        #
        #     # 将前60个添加到first_60_list
        #     assign_set.extend(block[:50])
        #
        #     # 将后40个添加到last_40_list
        #     eval_set.extend(block[50:])

        # 2. 用 assign_set 建 cost matrix + 匈牙利匹配（与原逻辑完全一致）
        cost_globals = {}
        for pred_pose_list, gt_pose, cids in assign_set:
            pred_point   = pred_pose_list[-1]['point'][0]
            pred_visible = pred_pose_list[-1]['visible'][0]
            gt_point     = gt_pose['point'][0]
            gt_visible   = gt_pose['visible'][0]

            cost_v  = (pred_visible.unsqueeze(1) >= score_th) & gt_visible.unsqueeze(0)
            cost_l2 = (pred_point.unsqueeze(1) - gt_point.unsqueeze(0)).square().sum(-1).sqrt() <= point_th
            cost_per = 1 - (cost_v & cost_l2).float()
            cid = cids[0].item()
            if cid in cost_globals:
                cost_globals[cid] = cost_globals[cid] + cost_per
            else:
                cost_globals[cid] = cost_per

        cid_2_m2k = {}
        for cid, cost_global in cost_globals.items():
            meta_idx, kp_idx = linear_sum_assignment(cost_global.cpu())
            reorder = kp_idx.argsort()
            cid_2_m2k[cid] = meta_idx[reorder]

        # Compute PR,RR for each class
        point_PRs, point_RRs, link_PRs, link_RRs = {}, {}, {}, {}
        for cid in cid_2_m2k.keys():
            point_PRs[cid] = []
            point_RRs[cid] = []
            link_PRs[cid] = []
            link_RRs[cid] = []
        for pred_pose_list, gt_pose, cids in eval_set:
            pred_point = pred_pose_list[-1]['point'][0]
            pred_visible = pred_pose_list[-1]['visible'][0]
            pred_link = pred_pose_list[-1]['link'][0]
            gt_point = gt_pose['point'][0]
            gt_visible = gt_pose['visible'][0]
            gt_link = gt_pose['link'][0]
            cid = cids[0].item()
            m2k = cid_2_m2k[cid]
            l2_condition = (pred_point[m2k][gt_visible] - gt_point[gt_visible]).square().sum(-1).sqrt() <= point_th
            vis_condition = pred_visible[m2k][gt_visible] >= score_th

            pred_visible_num = (pred_visible > score_th).sum()
            gt_visible_num = gt_visible.sum()

            point_hit = (l2_condition & vis_condition).sum()
            point_PR = point_hit / pred_visible_num if pred_visible_num != 0 else (pred_visible_num + 1)
            point_RR = point_hit / gt_visible_num
            point_PRs[cid].append(point_PR)
            point_RRs[cid].append(point_RR)

            gt_m = gt_link[gt_visible][:, gt_visible]
            target_link = torch.zeros_like(pred_link).bool()
            m2kv = m2k[gt_visible.cpu()]
            for x, y in zip(*torch.where(gt_m)):
                target_link[m2kv[x.item()], m2kv[y.item()]] = True
            assert (target_link[m2kv][:, m2kv] == gt_m).min().item()
            pred_link_dual = (pred_link + pred_link.T) / 2

            gt_link_num = target_link.triu(1).sum()
            pred_link_num = (pred_link_dual > score_th).triu(1).sum()
            link_hit = (pred_link_dual[target_link.triu(1)] >= score_th).sum()
            link_PR = link_hit / pred_link_num if pred_link_num != 0 else (pred_link_num + 1)
            link_RR = link_hit / gt_link_num
            link_PRs[cid].append(link_PR)
            link_RRs[cid].append(link_RR)
            if gt_link_num == 0 or link_hit == 0:
                d = 1
        return point_PRs, point_RRs, link_PRs, link_RRs

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        results = self.test_fn(runner.model, self.dataloader)
        filtered = [r for r in results if r is not None]
        if len(filtered) != 0:
            results = filtered
        else:
            return
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        pths = self.pths
        sths = self.sths
        point_mAPs = []
        link_mAPs = []
        point_AP_classes = []
        link_AP_classes = []
        pth_sth_map = {}
        for pth in pths:
            pPRs, pRRs, lPRs, lRRs = [], [], [], []
            for sth in sths:
                if sth == 0.5 and pth == 0.1:
                    epoch = runner.epoch
                else:
                    epoch = None
                # point_PRs, point_RRs, link_PRs, link_RRs = self.get_metric_given_valid_th(results, pth, sth, epoch)
                point_PRs, point_RRs, link_PRs, link_RRs = self.get_metric_given_valid_th(results, pth, sth, epoch)
                pPRs.append(point_PRs)
                pRRs.append(point_RRs)
                lPRs.append(link_PRs)
                lRRs.append(link_RRs)
            cids = point_PRs.keys()
            point_AP_per_class = []
            link_AP_per_class = []
            for cid in cids:
                point_PRs_for_imgs = torch.stack([torch.stack(item[cid]) for item in pPRs]).T
                point_RRs_for_imgs = torch.stack([torch.stack(item[cid]) for item in pRRs]).T
                pAPs = []
                for img_idx in range(point_RRs_for_imgs.size(0)):
                    prs = point_PRs_for_imgs[img_idx]
                    rrs = point_RRs_for_imgs[img_idx]
                    pAPs.append(compute_ap(prs, rrs))
                point_AP = torch.stack(pAPs).mean()
                point_AP_per_class.append(point_AP)



                link_PRs_for_imgs = torch.stack([torch.stack(item[cid]) for item in lPRs]).T
                link_RRs_for_imgs = torch.stack([torch.stack(item[cid]) for item in lRRs]).T
                lAPs = []
                for img_idx in range(point_RRs_for_imgs.size(0)):
                    prs = link_PRs_for_imgs[img_idx]
                    rrs = link_RRs_for_imgs[img_idx]
                    lAPs.append(compute_ap(prs, rrs))
                link_AP = torch.stack(lAPs).mean()
                link_AP_per_class.append(link_AP)

            point_AP_per_class = torch.stack(point_AP_per_class)
            link_AP_per_class = torch.stack(link_AP_per_class)
            point_mAPs.append(point_AP_per_class.nanmean())
            link_mAPs.append(link_AP_per_class.nanmean())
            point_AP_classes.append(point_AP_per_class)
            link_AP_classes.append(link_AP_per_class)

        point_mAP = torch.stack(point_mAPs).mean().item()
        link_mAP = torch.stack(link_mAPs).mean().item()

        pose_mAP = (point_mAP + link_mAP) / 2
        history_item = {'Epoch': runner.epoch}
        history_item['pose_mAP'] = pose_mAP
        history_item['mAP'] = point_mAP
        history_item['link_mAP'] = link_mAP
        for k, v in history_item.items():
            runner.log_buffer.output[f'{k}'] = v
        runner.log_buffer.ready = True

        history_item['point_mAPs'] = torch.stack(point_mAPs).cpu()
        history_item['link_mAPs'] = torch.stack(link_mAPs).cpu()

        history_item['point_AP_classes'] = torch.stack(point_AP_classes).mean(0)
        history_item['link_AP_classes'] = torch.stack(link_AP_classes).mean(0)

        self.metrics_history.append(history_item)
        self.log_recent_metric(runner.logger.info)

        key_score = history_item[self.key_indicator]
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)
        return

    def evaluate(self, runner, results):
        raise NotImplementedError()
        return

    def log_recent_metric(self, printf=print):
        if len(self.metrics_history) == 0:
            return
        else:
            history_item = self.metrics_history[-1]
            printf('')
            printf('--------------------------------------')
            printf('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
            printf(f'Metrics at Epoch {history_item["Epoch"]}:')
            printf(f'\tPose mAP:\t{history_item["pose_mAP"]:.1%};')
            printf(f'\tPoint mAP:\t{history_item["mAP"]:.1%};')
            printf(f'\tLink mAP:\t{history_item["link_mAP"]:.1%};')
            printf('')
            printf(f'Point Dist Thresholds: {self.pths}')
            printf(f'\tPoint mAPs:\t{history_item["point_mAPs"]};')
            printf(f'\tLink mAPs:\t{history_item["link_mAPs"]};')
            printf('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            printf('--------------------------------------')
            printf('')
            return


def compute_ap(PRs, RRs):
    ap = 0
    for i in range(len(PRs)):
        if i == 0:
            width = RRs[i]
        else:
            width = RRs[i] - RRs[i - 1]
        height = PRs[i:].max()
        ap += width * height
    return ap


class MSDistEvalHook(_DistEvalHook):
    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 test_fn=None,
                 greater_keys=MMPOSE_GREATER_KEYS,
                 less_keys=MMPOSE_LESS_KEYS,
                 broadcast_bn_buffer=True,
                 tmpdir=None,
                 gpu_collect=False,
                 **eval_kwargs):

        if test_fn is None:
            from mmpose.apis import multi_gpu_test
            test_fn = multi_gpu_test
        # to be compatible with the config before v0.16.0
        # update "save_best" according to "key_indicator" and remove the
        # latter from eval_kwargs
        if 'key_indicator' in eval_kwargs or isinstance(save_best, bool):
            warnings.warn(
                '"key_indicator" will be deprecated in EvalHook.'
                'Please use "save_best" to specify the metric key,'
                'e.g., save_best="AP".', DeprecationWarning)

            key_indicator = eval_kwargs.pop('key_indicator', 'AP')
            if save_best is True and key_indicator is None:
                raise ValueError('key_indicator should not be None, when '
                                 'save_best is set to True.')
            save_best = key_indicator
        super().__init__(dataloader, start, interval, by_epoch, save_best,
                         rule, test_fn, greater_keys, less_keys,
                         broadcast_bn_buffer, tmpdir, gpu_collect,
                         **eval_kwargs)

    def _do_evaluate(self, runner):
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = self.test_fn(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)

        if results is None:
            return
        filtered = [r for r in results if r is not None]
        if len(filtered) != 0:
            results = filtered
        else:
            return

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            key_score = self.evaluate(runner, [r['major'] for r in results], respostfix='major')
            aux_keys = ['prop'] + [f'ref{i}' for i in range(10)]
            aux_keys += [f'meta_{i}' for i in range(30)]
            aux_keys += [f'refine_{i}' for i in range(30)]
            for ak in aux_keys:
                if ak not in results[0]:
                    continue
                aux_score = self.evaluate(runner, [r[ak] for r in results], respostfix=ak)

            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)

    def evaluate(self, runner, results, respostfix=''):
        for key in ['preds', 'boxes', 'image_paths', 'bbox_ids']:
            assert key in results[0]

        eval_res = self.dataloader.dataset.evaluate(
            results, respostfix=respostfix, logger=runner.logger, **self.eval_kwargs)

        runner.logger.info('-----------------------------------------------')
        for name, val in eval_res.items():
            runner.log_buffer.output[f'{respostfix}-{name}'] = val
            if 'PCK' in name:
                runner.logger.info(f'\t\t{respostfix}-{name}: {val:.1%}')
            else:
                runner.logger.info(f'\t\t{respostfix}-{name}: {val:.3f}')
        runner.log_buffer.ready = True
        if self.save_best is not None:
            if not eval_res:
                warnings.warn('Since `eval_res` is an empty dict, the behavior to save '
                              'the best checkpoint will be skipped in this evaluation.')
                return None
            if self.key_indicator == 'auto':
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]
        else:
            return None
