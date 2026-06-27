from mmpose.datasets import DATASETS
import random
import numpy as np
import os
from collections import OrderedDict
from xtcocotools.coco import COCO
from capeformer.datasets.datasets.mp100.test_dataset import TestPoseDataset
import json_tricks as json
from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe, keypoint_nme,
                                                  keypoint_pck_accuracy)


@DATASETS.register_module()
class TestPoseORDataset(TestPoseDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 valid_class_ids,
                 max_kpt_num=None,
                 num_shots=1,
                 num_queries=100,
                 num_episodes=1,
                 pck_threshold_list=[0.05, 0.1, 0.15, 0.20],
                 test_mode=True):
        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            valid_class_ids,
            max_kpt_num,
            num_shots,
            num_queries,
            num_episodes,
            pck_threshold_list,
            test_mode)

        # self.replace_viz_pairs_1shot()
        return

    def replace_viz_pairs_1shot(self):
        # in [s,q] order
        self.paired_samples = [
            [1029, 1001],
            [1026, 1054],
            [1029, 1075],
            [1064, 1085],
            [1585, 1501],
            [1518, 1516],
            [1553, 1524],
            [1541, 1563],
            [1162, 1100],
            [480, 412],
            [403, 415],
            [452, 415],
            [450, 421],
            [467, 441],
            [489, 494],
            [1453, 1438],
            [1340, 1319],
            [1357, 1308],
            [1339, 1357],
            [1354, 1357],
            [1380, 1374],
            [789, 725],
            [790, 731],
            [722, 761],
        ]
        return

    def _report_metric_given(self, metrics, outputs, gts, masks, threshold_bbox):
        info_str = []

        if 'PCK' in metrics:
            pck_results = dict()
            for pck_thr in self.PCK_threshold_list:
                pck_results[pck_thr] = []

            for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks, threshold_bbox):
                for pck_thr in self.PCK_threshold_list:
                    _, pck, _ = keypoint_pck_accuracy(np.expand_dims(output, 0), np.expand_dims(gt, 0),
                                                      np.expand_dims(mask, 0), pck_thr, np.expand_dims(thr_bbox, 0))
                    pck_results[pck_thr].append(pck)

            mPCK = 0
            for pck_thr in self.PCK_threshold_list:
                info_str.append(['PCK@' + str(pck_thr), np.mean(pck_results[pck_thr])])
                mPCK += np.mean(pck_results[pck_thr])
            info_str.append(['mPCK', mPCK / len(self.PCK_threshold_list)])

        if 'NME' in metrics:
            nme_results = []
            for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks, threshold_bbox):
                nme = keypoint_nme(np.expand_dims(output, 0), np.expand_dims(gt, 0), np.expand_dims(mask, 0),
                                   np.expand_dims(thr_bbox, 0))
                nme_results.append(nme)
            info_str.append(['NME', np.mean(nme_results)])

        if 'AUC' in metrics:
            auc_results = []
            for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks, threshold_bbox):
                auc = keypoint_auc(np.expand_dims(output, 0), np.expand_dims(gt, 0), np.expand_dims(mask, 0),
                                   thr_bbox[0])
                auc_results.append(auc)
            info_str.append(['AUC', np.mean(auc_results)])

        if 'EPE' in metrics:
            epe_results = []
            for (output, gt, mask) in zip(outputs, gts, masks):
                epe = keypoint_epe(np.expand_dims(output, 0), np.expand_dims(gt, 0), np.expand_dims(mask, 0))
                epe_results.append(epe)
            info_str.append(['EPE', np.mean(epe_results)])
        return info_str

    def _report_metric(self, res_file, metrics):
        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.paired_samples)

        outputs = []
        gts = []
        masks = []
        threshold_bbox = []
        masks_or = []
        for pred, pair in zip(preds, self.paired_samples):
            item = self.db[pair[-1]]
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])

            mask_query = ((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            mask_sample = ((np.array(self.db[pair[0]]['joints_3d_visible'])[:, 0]) > 0)
            for id_s in pair[:-1]:
                mask_sample = np.bitwise_and(mask_sample, ((np.array(self.db[id_s]['joints_3d_visible'])[:, 0]) > 0))
                mask_sample_or = np.bitwise_or(mask_sample, ((np.array(self.db[id_s]['joints_3d_visible'])[:, 0]) > 0))

            masks.append(np.bitwise_and(mask_query, mask_sample))
            masks_or.append(np.bitwise_and(mask_query, mask_sample_or))

            if 'PCK' in metrics or 'NME' in metrics or 'AUC' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))

        info_str = self._report_metric_given(metrics, outputs, gts, masks, threshold_bbox)

        if self.num_shots > 1:
            info_str_or = self._report_metric_given(metrics, outputs, gts, masks_or, threshold_bbox)
            for k, v in info_str_or:
                info_str.append([f'or{k}', v])

        return info_str
