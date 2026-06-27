# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings

from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm
import os.path as osp
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
import cv2
import torch
import numpy as np
from vkits.vpose import *

MMPOSE_GREATER_KEYS = [
    'acc', 'ap', 'ar', 'pck', 'auc', '3dpck', 'p-3dpck', '3dauc', 'p-3dauc',
    'pcp', 'pose_mAP'
]
MMPOSE_LESS_KEYS = ['loss', 'epe', 'nme', 'mpjpe', 'p-mpjpe', 'n-mpjpe']


def matrix_to_upper_triangle_flat(matrix):
    """
    将一个二维方阵或三维批处理方阵的上三角部分（含对角线）提取并展平为向量。
    支持 NumPy 数组和 PyTorch 张量。
    """
    if isinstance(matrix, np.ndarray):
        # 处理 NumPy 数组
        if matrix.ndim == 2:
            n = matrix.shape[0]
            indices = np.triu_indices(n, k=0)
            return matrix[indices[0], indices[1]]
        elif matrix.ndim == 3:
            b, n, _ = matrix.shape
            indices = np.triu_indices(n, k=0)
            return matrix[:, indices[0], indices[1]]
        else:
            raise ValueError("输入必须是2D或3D的方阵")

    elif torch.is_tensor(matrix):
        # 处理 PyTorch 张量
        device = matrix.device
        if matrix.dim() == 2:
            n = matrix.size(0)
            indices = torch.triu_indices(n, n, offset=0, device=device)
            return matrix[indices[0], indices[1]]
        elif matrix.dim() == 3:
            b, n, _ = matrix.size()
            indices = torch.triu_indices(n, n, offset=0, device=device)
            return matrix[:, indices[0], indices[1]]
        else:
            raise ValueError("输入必须是2D或3D的方阵")

    else:
        raise TypeError("输入必须是 NumPy 数组或 PyTorch 张量")


def draw_gt_pose(marker_factor=0.03, font_factor=0.5, **kwargs):
    img_t = kwargs['img']
    if isinstance(img_t, np.ndarray):
        img_n = img_t
    else:
        mean = torch.tensor([0.485, 0.456, 0.406])[None, None, :].to(img_t)
        std = torch.tensor([0.229, 0.224, 0.225])[None, None, :].to(img_t)
        img_n = (img_t.permute(1, 2, 0) * std + mean).flip(-1).cpu().numpy() * 255

    img_h, img_w, _ = img_n.shape
    img_n = img_n.astype(np.uint8)

    img = copy.deepcopy(img_n).astype(np.uint8)
    img = np.ascontiguousarray(img)
    radius = max(int(min(img_h, img_w) * marker_factor), 1)
    red = [0, 0, 1]
    blue = [1, 0, 0]
    link_width = radius // 2

    points = (kwargs['points'] * img_h).long()
    visibles = kwargs['visibles']
    # link_matrix = kwargs['links']
    #
    # link_color = [255, 255, 0]
    # vl_matrix = link_matrix[visibles][:, visibles]
    # vpoints = points[visibles]
    # V = len(vpoints)
    # for i in range(V):
    #     for j in range(V):
    #         if i > j and vl_matrix[i, j] == 1:
    #             cv2.line(img, vpoints[i].tolist(), vpoints[j].tolist(), link_color, link_width)

    max_v = torch.where(visibles)[0].max().item() + 1
    point_colors = get_color_list_hsv(red, blue, max_v)
    for n in range(max_v):
        if visibles[n].bool():
            r, g, b = point_colors[n]
            color = (int(r * 255), int(g * 255), int(b * 255))
            x_coord, y_coord = points[n].tolist()
            cv2.circle(img, (x_coord, y_coord), radius, color, -1)
            if kwargs.get('show_kid', False):
                cv2.putText(img, f'P{n}',
                            (x_coord, y_coord),
                            cv2.FONT_HERSHEY_COMPLEX, 1, [1, 0, 0], 2)

    if 'fpath' in kwargs:
        cv2.imwrite(kwargs['fpath'], img)
        return
    else:
        return img


# 进行hook l2指标的运算 在原有基础上拟合test数据的W

class L2_W_EvalHook(_EvalHook):

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
        # self.pths = [0.05, 0.1, 0.15, 0.2]
        self.pths = [0.05]
        self.sths = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        return

    def evaluate_link_performance_symmetric(self, test_data, link_regressor):
        """
        在测试数据上评估对称优化的连接回归器。
        """
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        num_evaluated = 0
        device = link_regressor.device

        # 预先计算索引（只计算一次）
        gt_triu_indices = torch.triu_indices(70, 70, device=device)

        for pred_pose_list, gt_pose, cids, _ in test_data:
            pred_link = pred_pose_list[-1]['link'][0]  # [100, 100]
            gt_link = gt_pose['link'][0]  # [70, 70]
            gt_visible = gt_pose['visible'][0]  # [70]

            # 1. 将输入和输出矩阵转换为上三角向量
            pred_link_upper_flat = matrix_to_upper_triangle_flat(pred_link)  # [5050]
            gt_link_upper_flat = matrix_to_upper_triangle_flat(gt_link)  # [2485]

            # 2. 使用回归器预测
            estimated_gt_link_upper_flat = pred_link_upper_flat @ link_regressor  # [2485]

            # 3. 创建可见性掩码
            gt_i_indices = gt_triu_indices[0]
            gt_j_indices = gt_triu_indices[1]
            link_visibility_mask = (gt_visible[gt_i_indices] == 1) & (gt_visible[gt_j_indices] == 1)

            # 4. 只在可见的连接上计算误差
            visible_count = link_visibility_mask.sum().item()
            if visible_count > 0:
                visible_pred = estimated_gt_link_upper_flat[link_visibility_mask]
                visible_gt = gt_link_upper_flat[link_visibility_mask]

                mse = torch.nn.functional.mse_loss(visible_pred, visible_gt)
                mae = torch.nn.functional.l1_loss(visible_pred, visible_gt)

                total_mse += mse * visible_count
                total_mae += mae * visible_count
                total_samples += visible_count
                num_evaluated += 1

        # 计算加权平均值并转换为 Python 原生类型
        if total_samples > 0:
            avg_mse = (total_mse / total_samples).item()
            avg_mae = (total_mae / total_samples).item()
        else:
            avg_mse = 0.0
            avg_mae = 0.0

        return {
            "mse": avg_mse,
            "mae": avg_mae,
            "evaluated_samples": num_evaluated,
            "total_connections": total_samples
        }

    def return_link_regressor_symmetric(self, pred_links, gt_links, gt_visibles):
        """
        计算利用了对称性的连接回归器 W_link。
        输入和输出都是矩阵的上三角部分。
        此版本经过加固，可以处理某些连接在所有样本中都不可见的情况。

        Args:
            pred_links (np.ndarray): 堆叠的预测连接矩阵, [样本数, 100, 100]
            gt_links (np.ndarray): 堆叠的真实连接矩阵, [样本数, 70, 70]
            gt_visibles (np.ndarray): 堆叠的真实可见性向量, [样本数, 70]

        Returns:
            np.ndarray: 优化后的回归器 W_link，形状约为 [5050, 2485]
        """
        num_samples, num_pred_kpts, _ = pred_links.shape
        _, num_gt_kpts, _ = gt_links.shape

        # 1. 将输入和输出矩阵转换为上三角向量
        # 确保输入是 NumPy array
        X = matrix_to_upper_triangle_flat(np.asarray(pred_links))  # 形状: [样本数, 5050]
        Y = matrix_to_upper_triangle_flat(np.asarray(gt_links))  # 形状: [样本数, 2485]

        # 确保可见性也是 NumPy array
        gt_visibles = np.asarray(gt_visibles)

        # 初始化最终的回归器矩阵
        W_link = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float64)
        print(f"开始训练对称优化的连接回归器，输入维度: {X.shape[1]}, 目标值数量: {Y.shape[1]}...")

        # 预先计算上三角索引，用于从 k 映射回 (i, j)
        # triu_indices 返回的是 (行索引数组, 列索引数组)
        gt_triu_indices = np.triu_indices(num_gt_kpts, 0)

        # 2. 遍历展平后的上三角向量的每一个目标列
        for k in range(Y.shape[1]):
            print(k)
            if k % 100 == 0 or k == Y.shape[1] - 1:
                print(f"正在处理目标值 {k + 1}/{Y.shape[1]}...")

            # a. 确定这个目标值对应的真实关键点索引 (i, j)
            gt_i = gt_triu_indices[0][k]
            gt_j = gt_triu_indices[1][k]

            # b. 找到两个端点都可见的样本
            link_visible_mask = (gt_visibles[:, gt_i] == 1) & (gt_visibles[:, gt_j] == 1)
            visible_indices = np.where(link_visible_mask)[0]

            # c. 【核心修复】只有在找到了至少一个有效样本时，才进行后续操作
            if visible_indices.size > 0:
                try:
                    # 过滤 X 和 Y
                    X_filtered = X[visible_indices, :]
                    Y_filtered_k = Y[visible_indices, k]

                    # d. 求解权重（即 W_link 的第 k 列）
                    W_k = np.linalg.pinv(X_filtered.T @ X_filtered) @ X_filtered.T @ Y_filtered_k
                    W_link[:, k] = W_k

                except np.linalg.LinAlgError:
                    # 如果在计算过程中出现线性代数错误（例如矩阵奇异），则捕获并报告
                    print(f"警告: SVD 算法在处理目标值 {k} (对应连接 {gt_i}-{gt_j}) 时未收敛，已跳过。")
                    # W_link[:, k] 将保持为全零
                    continue
            # else:
            #   如果 visible_indices.size == 0，则什么也不做。
            #   W_link 的第 k 列将保持为初始化的零向量，然后循环继续到下一个 k。

        print("对称优化连接回归器训练完成。")
        return W_link



    def return_regressor_visible(self,X, Y, visible):
        # X[29, 20]  Y[29,30] visible[29,30]
        import numpy as np

        # find mean of X
        # X[:, :, :2] -= 0.5
        Y = Y - 0.5

        # Initialize W to have the same number of columns as keypoints
        W = np.zeros((X.shape[1], Y.shape[1]))

        # Iterate through each keypoint
        for j in range(Y.shape[1]):
            # Indices where this keypoint is visible
            visible_indices = np.where(visible[:, j] == 1)[0]  # [number_of_visible_samples_for_j]

            # Filter X and Y matrices based on visibility of this keypoint
            X_filtered = X[visible_indices, :]  # [number_of_visible_samples_for_j, 20]
            Y_filtered = Y[visible_indices, j]  # [number_of_visible_samples_for_j]

            # Solve for the weights related to this keypoint
            W_j = np.linalg.pinv(X_filtered.T @ X_filtered) @ X_filtered.T @ Y_filtered  # [300]

            # Store these weights in the W matrix
            W[:, j] = W_j

        return W
    def _distance_acc(self, distances, thr=0.10):

        distance_valid = distances != -1
        num_distance_valid = distance_valid.sum()
        if num_distance_valid > 0:
            return distances[distance_valid].sum() / num_distance_valid
        return -1

    def global_pck(self, results, thr):

        distances_one = np.full((len(results), len(results[0][1])), -1, dtype=np.float32)

        for b, (pred_point, gt_point, gt_visible,cids) in enumerate(results):

            cid = cids.item()
            pred_point, gt_point, gt_visible, cids = \
                (pred_point.detach().cpu().numpy(), gt_point.detach().cpu().numpy(),
                 gt_visible.detach().cpu().numpy(), cids.detach().cpu().numpy(),

                 )

            distances_one[b][gt_visible] = np.linalg.norm((pred_point[gt_visible] - gt_point[gt_visible]), axis=-1)

        acc_one = np.array([self._distance_acc(d, thr) for d in distances_one])
        valid_acc_one = acc_one[acc_one >= 0]
        cnt_one = len(valid_acc_one)

        avg_pck_one = valid_acc_one.mean() if cnt_one > 0 else 0

        return avg_pck_one


    def mean_l2(self, results, thr,regressor):
        distances = []
        for b, (pred_point, pred_point_visible,gt_point, gt_visible,cids) in enumerate(results):
            estimated_kpts = torch.cat((pred_point -0.5, pred_point_visible.unsqueeze(1)), dim=1)
            estimated_kpts = ((estimated_kpts.view(1, -1)) @ regressor) + 0.5
            estimated_kpts = estimated_kpts.view(-1, 2)
            l2 = (estimated_kpts - gt_point).norm(dim=-1)
            l2_mean = (l2 * gt_visible).sum()
            l2_mean /= gt_visible.sum()
            distances.append(l2_mean.cpu())

        return torch.mean(torch.stack(distances))

    def mean_l2_and_pck(self, results, thr, regressor,foward = None):
        distances_l2 = []
        pck_256 = []
        pck_th = []
        for b, (pred_pose_list, gt_pose, cids,img_n) in enumerate(results):

            pred_point = pred_pose_list[-1]['point'][0]
            pred_point_visible = pred_pose_list[-1]['visible'][0]
            gt_point = gt_pose['point'][0]
            gt_visible = gt_pose['visible'][0].int()
            pred_visible = torch.ones_like(gt_visible)

            estimated_kpts = torch.cat((pred_point - 0.5, pred_point_visible.unsqueeze(1)), dim=1)
            estimated_kpts = ((estimated_kpts.view(1, -1)) @ regressor) + 0.5
            estimated_kpts = estimated_kpts.view(-1, 2)

            if b % 100 == 0:
                imged_pred = draw_gt_pose(img=img_n, points=estimated_kpts,
                                        visibles=pred_visible)
                imaged_GT = draw_gt_pose(img=img_n, points=gt_point,
                                          visibles=gt_visible)

                row0 = {'IMG': img_n / 255, 'GT': imaged_GT}
                row1 = {'IMG': img_n / 255, 'Pred': imged_pred}
                if foward:
                    viz_dict_list([row0,row1], fpath=f'foward_{b}.jpg')
                else:
                    viz_dict_list([row0, row1], fpath=f'test_{b}.jpg')

            # l2
            l2 = (estimated_kpts - gt_point).norm(dim=-1)
            l2_mean = (l2 * gt_visible).sum()
            l2_mean /= gt_visible.sum()
            distances_l2.append(l2_mean.cpu())

            # pck0.2
            pck_t= (estimated_kpts - gt_point).norm(dim=-1)
            pck_t = ((pck_t < 0.2).float()*gt_visible).sum()
            pck_t /= gt_visible.sum()
            pck_th.append(pck_t.cpu())


            # pck
            estimated_kpts *= 256
            gt_point *= 256
            l2_pck = (estimated_kpts - gt_point).norm(dim=-1)
            pck_mean = ((l2_pck < 6).float()*gt_visible).sum()
            pck_mean /= gt_visible.sum()
            pck_256.append(pck_mean.cpu())



        return torch.mean(torch.stack(distances_l2)),torch.mean(torch.stack(pck_256)),torch.mean(torch.stack(pck_th))



    def global_l2(self, results, thr):

        distances_one = np.full((len(results), len(results[0][1])), -1, dtype=np.float32)

        for b, (pred_point, gt_point, gt_visible,cids) in enumerate(results):

            cid = cids.item()
            pred_point, gt_point, gt_visible, cids = \
                (pred_point.detach().cpu().numpy(), gt_point.detach().cpu().numpy(),
                 gt_visible.detach().cpu().numpy(), cids.detach().cpu().numpy(),

                 )

            distances_one[b][gt_visible] = np.linalg.norm((pred_point[gt_visible] - gt_point[gt_visible]), axis=-1)

        acc_one = np.array([self._distance_acc(d, thr) for d in distances_one])
        valid_acc_one = acc_one[acc_one >= 0]
        cnt_one = len(valid_acc_one)

        avg_pck_one = acc_one.mean() if cnt_one > 0 else 0

        return avg_pck_one



    def get_metric_given_valid_th_singlestage(self, results, point_th, score_th, epoch=None):
        # Global Asignment with the Last Prediction
        cost_globals = {}
        # cost_total = torch.zeros(70,70).to("cuda")
        # for pred_pose_list, gt_pose, cids, iids in results:
        for pred_pose_list, gt_pose, cids in results:
            pred_point = pred_pose_list[-1]['point'][0]
            pred_visible = pred_pose_list[-1]['visible'][0]
            gt_point = gt_pose['point'][0]
            gt_visible = gt_pose['visible'][0]
            # 当预测的关键点和真实关键点都为True时，cost_v为True
            cost_v = (pred_visible.unsqueeze(1) >= score_th) & gt_visible.unsqueeze(0)
            # 距离是否在阈值内，若处于阈值内则为True
            cost_l2 = (pred_point.unsqueeze(1) - gt_point.unsqueeze(0)).square().sum(-1).sqrt() <= point_th
            # 若满足条件预测可见对应真实可见，并且在阈值内，则cost_per 为0
            cost_per = 1 - (cost_v & cost_l2).float()
            # cost_total = cost_total + cost_per
            cid = cids[0].item()
            if cid in cost_globals:
                cost_globals[cid] = cost_globals[cid] + cost_per
            else:
                cost_globals[cid] = cost_per
        # # For all_test_data assignment
        # meta_idx_all, kp_idx_all = linear_sum_assignment(cost_total.cpu())
        # reorder = kp_idx_all.argsort()
        # meta_ridx_all = meta_idx_all[reorder]
        # kp_ridx_all = kp_idx_all[reorder]
        # global_pck = self.global_pck(results, meta_ridx_all, point_th)
        # Solve the assignment for each class
        cid_2_m2k = {}
        for cid, cost_global in cost_globals.items():
            meta_idx, kp_idx = linear_sum_assignment(cost_global.cpu())
            reorder = kp_idx.argsort()
            meta_ridx = meta_idx[reorder]
            kp_ridx = kp_idx[reorder]
            cid_2_m2k[cid] = meta_ridx
        global_pck = self.global_singlestage_pck(results, cid_2_m2k, point_th)
        # Compute PR,RR for each class
        # point_PRs, point_RRs, link_PRs, link_RRs = {}, {}, {}, {}
        point_PRs, point_RRs = {}, {}
        for cid in cid_2_m2k.keys():
            point_PRs[cid] = []
            point_RRs[cid] = []
            # link_PRs[cid] = []
            # link_RRs[cid] = []
        # for r, (pred_pose_list, gt_pose, cids, iids) in enumerate(results):
        for r, (pred_pose_list, gt_pose, cidsglag) in enumerate(results):
            pred_point = pred_pose_list[-1]['point'][0]
            pred_visible = pred_pose_list[-1]['visible'][0]
            # pred_link = pred_pose_list[-1]['link'][0]
            gt_point = gt_pose['point'][0]
            gt_visible = gt_pose['visible'][0]
            # gt_link = gt_pose['link'][0]
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

        return point_PRs, point_RRs, global_pck

    def get_metric_given_valid_th(self, results, point_th, score_th, epoch=None):
        cid_list = []
        source_keypoints = []
        target_keypoints = []
        target_visible = []
        # for pred_point, pred_point_visible,gt_point, visible,cids in results:
        #     cid = cids[0].item()
        #     cid_list.append(cid)
        # cid_list_tensor = torch.tensor(cid_list)
        # unique_cid = torch.unique(cid_list_tensor)'

        # 初始化两个列表分别存储前60个和后40个元素
        first_60_list = []  # 存储每100个中的前60个
        last_40_list = []  # 存储每100个中的后40个

        # 遍历results列表，步长为100
        for i in range(0, len(results), 100):
            # 获取当前100个元素的块
            block = results[i:i + 100]

            # 将前60个添加到first_60_list
            first_60_list.extend(block[:60])

            # 将后40个添加到last_40_list
            last_40_list.extend(block[60:])




        # path = os.path.join('weights/cub_muti_s1/ours_b3n30_muti_L50.pt')
        path = os.path.join('weights/test/forward60_no_order.pt')
        if  os.path.exists(path):
            regressor = torch.load(path).to(results[0][2].device)
        else :
            regressor = None
        if regressor is None:

            for pred_pose_list, gt_pose, cids,_ in first_60_list:
                pred_point = pred_pose_list[-1]['point'][0]
                pred_point_visible = pred_pose_list[-1]['visible'][0]
                gt_point = gt_pose['point'][0]
                visible = gt_pose['visible'][0].int()

                source_keypoints.append(torch.cat((pred_point, pred_point_visible.unsqueeze(1)), dim=1))
                target_keypoints.append(gt_point)
                target_visible.append(visible)
            source_keypoints = torch.stack(source_keypoints)
            target_keypoints = torch.stack(target_keypoints)
            target_visible = torch.stack(target_visible)
            source_keypoints[:, :, :2] -= 0.5
            visible_reshaped = target_visible.unsqueeze(-1).repeat(1, 1, 2).reshape(target_visible.shape[0],target_visible.shape[1] * 2)
            regressor = self.return_regressor_visible(
                source_keypoints.cpu().numpy().reshape(source_keypoints.shape[0], source_keypoints.shape[1] * 3).astype(np.float64),
                target_keypoints.cpu().numpy().reshape(target_keypoints.shape[0], target_keypoints.shape[1] * 2).astype(np.float64),
                visible_reshaped.cpu().numpy().astype(np.float64),
            )
            regressor = torch.tensor(regressor).to(torch.float32).to(source_keypoints.device)
            torch.save(regressor, path)

        # link
        # path_link = os.path.join('weights/link_weights/link_regressor_symmetric.pt')  # 使用新文件名
        path_link = os.path.join('weights/link_weights/link_regressor.pt')  # 使用新文件名
        if os.path.exists(path_link):
            regressor_link = torch.load(path_link).to(results[0][1]['point'][0].device)
        else:
            print("对称优化连接回归器未找到，开始训练...")
            pred_links_list, gt_links_list, gt_visibles_list = [], [], []

            for pred_pose_list, gt_pose, cids, _ in first_60_list:
                pred_links_list.append(pred_pose_list[-1]['link'][0].cpu().numpy())
                gt_links_list.append(gt_pose['link'][0].cpu().numpy())
                gt_visibles_list.append(gt_pose['visible'][0].cpu().numpy())

            pred_links_np = np.stack(pred_links_list)
            gt_links_np = np.stack(gt_links_list)
            gt_visibles_np = np.stack(gt_visibles_list)

            # 调用新的训练函数
            W_link_np = self.return_link_regressor_symmetric(pred_links_np, gt_links_np, gt_visibles_np)

            regressor_link = torch.tensor(W_link_np, dtype=torch.float32).to(results[0][1]['point'][0].device)
            torch.save(regressor_link, path_link)









        forward_mean_l2, forward_mean_pck, forward_mean_pck_th = self.mean_l2_and_pck(first_60_list, point_th, regressor,foward=True)
        mean_l2,mean_pck,mean_pck_th = self.mean_l2_and_pck(last_40_list, point_th,regressor)
        link_metrics_first = self.evaluate_link_performance_symmetric(first_60_list, regressor_link)
        print("测试集上前60的连接评估指标 (对称优化):")
        print(f"  MSE: {link_metrics_first['mse']:.6f}")
        print(f"  MAE: {link_metrics_first['mae']:.6f}")
        print(
            f"  评估样本数: {link_metrics_first['evaluated_samples']}, 总连接数: {link_metrics_first['total_connections']}")
        link_metrics_last = self.evaluate_link_performance_symmetric(last_40_list, regressor_link)
        print("测试集上后40的连接评估指标 (对称优化):")
        print(f"  MSE: {link_metrics_last['mse']:.6f}")
        print(f"  MAE: {link_metrics_last['mae']:.6f}")
        print(
            f"  评估样本数: {link_metrics_last['evaluated_samples']}, 总连接数: {link_metrics_last['total_connections']}")

        # global_pck_one = self.global_l2(results, point_th)


        return forward_mean_l2, forward_mean_pck, forward_mean_pck_th,mean_l2,mean_pck,mean_pck_th

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

        pmPck_one = {}

        sdkp_pck = {}
        pck_twenty ={}
        forward_pmPck_one = {}
        forward_sdkp_pck = {}
        forward_pck_twenty = {}


        for pth in pths:
            # pPRs, pRRs, lPRs, lRRs = [], [], [], []
            pPRs_one, pRRs_one, pPRs_two, pRRs_two = [], [], [], []
            sth_pck_one = []
            # 这两个是自己加的
            sth_pck_two = []
            sth_pck_three = []
            forward_sth_pck_one = []
            forward_sth_pck_two = []
            forward_sth_pck_three = []

            sth = 0
            epoch = None
            forward_mean_l2, forward_mean_pck, forward_mean_pck_th,mean_l2,mean_pck,mean_pck_th = self.get_metric_given_valid_th(                    results,
                    pth, sth, epoch)

            sth_pck_one.append(mean_l2.item())
            sth_pck_two.append(mean_pck.item())
            sth_pck_three.append(mean_pck_th.item())
            forward_sth_pck_one.append(forward_mean_l2.item())
            forward_sth_pck_two.append(forward_mean_pck.item())
            forward_sth_pck_three.append(forward_mean_pck_th.item())


            pmPck_one[pth] = sth_pck_one
            sdkp_pck[pth] = sth_pck_two
            pck_twenty[pth] = sth_pck_three
            forward_pmPck_one[pth] = forward_sth_pck_one
            forward_sdkp_pck[pth] = forward_sth_pck_two
            forward_pck_twenty[pth] = forward_sth_pck_three





        history_item = {'Epoch': runner.epoch}

        history_item['mean_l2'] = pmPck_one
        history_item['mean_pck'] = sdkp_pck
        history_item['pck_th0.2'] = pck_twenty

        history_item['forward_mean_l2'] = forward_pmPck_one
        history_item['forward_mean_pck'] = forward_sdkp_pck
        history_item['forward_pck_th0.2'] = forward_pck_twenty

        for k, v in history_item.items():
            runner.log_buffer.output[f'{k}'] = v
        runner.log_buffer.ready = True

        self.metrics_history.append(history_item)
        self.log_recent_metric(runner.logger.info)



    def log_recent_metric(self, printf=print):
        if len(self.metrics_history) == 0:
            return
        else:
            history_item = self.metrics_history[-1]

            printf('')
            printf(f'Point Dist Thresholds: {self.pths}')
            for pth in self.pths:
                printf(f'l2:\t {history_item["mean_l2"][pth]}\n')
                printf(f'pck_sdkp:\t {history_item["mean_pck"][pth]}\n')
                printf(f'pck_0.2:\t {history_item["pck_th0.2"][pth]}\n')

                printf(f'forward_l2:\t {history_item["forward_mean_l2"][pth]}\n')
                printf(f'forward_pck_sdkp:\t {history_item["forward_mean_pck"][pth]}\n')
                printf(f'forward_pck_0.2:\t {history_item["forward_pck_th0.2"][pth]}\n')



            printf('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            printf('--------------------------------------')
            printf('')
            return









def compute_ap(PRs, RRs):
    # lower score always leads to higher RR
    # 每一个小矩形，取横坐标差为宽，取该点及右侧最大值为高。
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
