from mmpose.datasets import DATASETS
import random
import numpy as np
import os
from collections import OrderedDict
from xtcocotools.coco import COCO
from capeformer.datasets.datasets.mp100.test_dataset import TestPoseDataset
import copy
import torch


@DATASETS.register_module()
class MyTestSet(TestPoseDataset):
    def __init__(self, ann_file, img_prefix, data_cfg, pipeline, valid_class_ids,
                 max_kpt_num=None, num_shots=1, num_queries=100, num_episodes=1,
                 pck_threshold_list=[0.05, 0.1, 0.15, 0.20], test_mode=True, img_split_ratio=None, ind_order=False):
        if img_split_ratio is not None:
            assert img_split_ratio > 0 and img_split_ratio < 1

        self.img_split_ratio = img_split_ratio
        super().__init__(ann_file=ann_file, img_prefix=img_prefix, data_cfg=data_cfg, pipeline=pipeline,
                         valid_class_ids=valid_class_ids, max_kpt_num=max_kpt_num, num_shots=num_shots,
                         num_queries=num_queries, num_episodes=num_episodes, pck_threshold_list=pck_threshold_list)
        if ind_order:
            # ind_order = np.stack([np.random.permutation(max_kpt_num) for _ in range(100)])
            # np.save(f'output/ind_order.npy', ind_order)
            self.point_order_dict = np.load('data/ind_order.npy')
        else:
            self.point_order_dict = np.stack(
                [np.linspace(0, max_kpt_num - 1, max_kpt_num).astype(np.int) for _ in range(100)])
        return

    def make_paired_samples(self):
        random.seed(1)
        np.random.seed(0)
        all_samples = []
        for cls in self.valid_class_ids:
            all_samples_per = self.cat2obj[cls]
            if self.img_split_ratio is not None:
                train_num = int(len(all_samples_per) * self.img_split_ratio)
                test_samples = all_samples_per[train_num:]
            else:
                test_samples = all_samples_per

            all_samples += test_samples
        self.paired_samples = np.array(all_samples)

    def random_paired_samples(self):
        raise NotImplementedError

    def get_single_data(self, obj_id):
        obj = copy.deepcopy(self.db[obj_id])
        obj['ann_info'] = copy.deepcopy(self.ann_info)
        ori_data = self.pipeline(obj)

        _, img_H, img_W = ori_data['img'].shape
        assert img_H == img_W
        img_meta = ori_data['img_metas'].data
        cid = img_meta['category_id']
        keypoint = torch.tensor(img_meta['joints_3d'][:, :2]) / img_H
        visible = torch.tensor(ori_data['target_weight']).bool().squeeze(1)
        se_pairs = self.cats[cid]['skeleton']
        link = torch.zeros(len(visible), len(visible))
        for s, e in se_pairs:
            link[s, e] = 1
            link[e, s] = 1

        point_order = self.point_order_dict[cid - 1]
        return ori_data['img'], keypoint[point_order], visible[point_order], link[point_order][:, point_order], img_meta
        # return ori_data['img'], keypoint, visible, link, img_meta

    def __getitem__(self, idx):
        did = self.paired_samples[idx]
        img_q, keypoint_q, visible_q, link_q, meta_q = self.get_single_data(did)

        data = dict()
        data['cid'] = meta_q['category_id']
        data['did'] = did
        data['dataset_seq_id'] = idx
        data.update(dict(img_q=img_q, point_q=keypoint_q, visible_q=visible_q, link_q=link_q))
        return data

    def get_item_by_did(self, did, img_only=True):
        if img_only:
            obj = copy.deepcopy(self.db[did])
            obj['ann_info'] = copy.deepcopy(self.ann_info)
            ori_data = self.pipeline(obj)
            return ori_data['img']
        else:
            obj = copy.deepcopy(self.db[did])
            obj['ann_info'] = copy.deepcopy(self.ann_info)
            ori_data = self.pipeline(obj)
            _, img_H, img_W = ori_data['img'].shape
            assert img_H == img_W
            img_meta = ori_data['img_metas'].data
            cid = img_meta['category_id']
            keypoint = torch.tensor(img_meta['joints_3d'][:, :2]) / img_H
            visible = torch.tensor(ori_data['target_weight']).bool().squeeze(1)
            se_pairs = self.cats[cid]['skeleton']
            link = torch.zeros(len(visible), len(visible))
            for s, e in se_pairs:
                link[s, e] = 1
                link[e, s] = 1

            point_order = self.point_order_dict[cid - 1]
            return ori_data['img'], keypoint[point_order], visible[point_order], link[point_order][:,
                                                                                 point_order], img_meta
