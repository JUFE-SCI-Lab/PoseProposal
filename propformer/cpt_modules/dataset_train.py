from mmpose.datasets import DATASETS
import random
import numpy as np
import os
from collections import OrderedDict
from xtcocotools.coco import COCO
from capeformer.datasets.datasets.mp100.transformer_dataset import TransformerPoseDataset
import copy
import torch


@DATASETS.register_module()
class MyTrainSet(TransformerPoseDataset):
    def __init__(self, ann_file, img_prefix, data_cfg, pipeline, valid_class_ids,
                 max_kpt_num=None, num_shots=1, num_queries=100, num_episodes=1, test_mode=False,
                 img_split_ratio=None, epoch_sample_num=None, ind_order=False):
        self.img_split_ratio = img_split_ratio
        self.epoch_sample_num = epoch_sample_num
        super().__init__(ann_file=ann_file, img_prefix=img_prefix, data_cfg=data_cfg, pipeline=pipeline,
                         valid_class_ids=valid_class_ids, max_kpt_num=max_kpt_num, num_shots=num_shots,
                         num_queries=num_queries, num_episodes=num_episodes, test_mode=test_mode)

        if ind_order:
            # ind_order = np.stack([np.random.permutation(max_kpt_num) for _ in range(100)])
            # np.save(f'output/ind_order.npy', ind_order)
            self.point_order_dict = np.load('data/ind_order.npy')
        else:
            self.point_order_dict = np.stack(
                [np.linspace(0, max_kpt_num - 1, max_kpt_num).astype(np.int) for _ in range(100)])
        return

    def random_paired_samples(self):
        # Will be called every epoch.
        num_datas = [len(self.cat2obj[self._class_to_ind[cls]]) for cls in self.valid_classes]
        if self.epoch_sample_num is None:
            samples_per_class_in_this_epoch = max(num_datas)
        else:
            samples_per_class_in_this_epoch = int(self.epoch_sample_num / len(self.valid_classes))

        all_samples = []
        for cls in self.valid_class_ids:
            all_samples_per = self.cat2obj[cls]
            if self.img_split_ratio is not None:
                train_num = max(int(len(all_samples_per) * self.img_split_ratio), self.num_shots + 1)
                train_samples = all_samples_per[:train_num]
            else:
                train_samples = all_samples_per
            for i in range(samples_per_class_in_this_epoch):
                shot = random.sample(train_samples, self.num_shots + 1)
                all_samples.append(shot)

        self.paired_samples = np.array(all_samples)
        np.random.shuffle(self.paired_samples)

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
            if visible[s] and visible[e]:
                link[s, e] = 1
                link[e, s] = 1

        point_order = self.point_order_dict[cid - 1]
        return ori_data['img'], keypoint[point_order], visible[point_order], link[point_order][:, point_order], \
               ori_data['target'][point_order], img_meta
        # return ori_data['img'], keypoint, visible, link, ori_data['target'], img_meta

    def __getitem__(self, idx):
        pair_ids = self.paired_samples[idx]
        assert len(pair_ids) == self.num_shots + 1
        imgs, points, visibles, links, cids, metas = [], [], [], [], [], []
        heatmaps = []
        for did in pair_ids:
            img, point, visible, link, heatmap, meta = self.get_single_data(did)
            imgs.append(img)
            points.append(point)
            visibles.append(visible)
            links.append(link)
            metas.append(meta)
            cids.append(meta['category_id'])
            heatmaps.append(torch.tensor(heatmap))

        data = dict()
        data['imgs'] = torch.stack(imgs)
        data['points'] = torch.stack(points)
        data['visibles'] = torch.stack(visibles)
        data['links'] = torch.stack(links)
        data['cid'] = torch.tensor(cids)
        # data['metas'] =metas
        if self.num_shots == 0:
            data['heatmap'] = torch.stack(heatmaps)
        return data
