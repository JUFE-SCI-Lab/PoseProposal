from mmpose.datasets import DATASETS
import random
import numpy as np
import os
from capeformer.datasets.datasets.mp100.test_dataset import TestPoseDataset


@DATASETS.register_module()
class CrossClassQSTestPoseDataset(TestPoseDataset):

    def __init__(self, ann_file, img_prefix, data_cfg, pipeline, valid_class_ids, max_kpt_num=None, num_shots=1,
                 num_queries=100, num_episodes=1, pck_threshold_list=[0.05, 0.1, 0.15, 0.20, 0.25], test_mode=True):

        super().__init__(ann_file, img_prefix, data_cfg, pipeline, valid_class_ids, max_kpt_num, num_shots,
                         num_queries, num_episodes, pck_threshold_list, test_mode)
        assert test_mode
        return

    def _set_qcid_to_scid(self):
        self.cross_class_qs_infos = self.coco.dataset['info']['cross_class_qs_infos']
        self.qcid_to_scid = {}
        for info in self.cross_class_qs_infos:
            self.qcid_to_scid[info['id']] = info['support_class_ids']
        return

    def make_paired_samples(self):
        self._set_qcid_to_scid()
        random.seed(1)
        np.random.seed(0)

        all_samples = []
        for qcid in self.valid_class_ids:
            for _ in range(self.num_episodes):
                scids = self.qcid_to_scid[qcid]
                if len(scids) == 0:
                    pass
                else:
                    scid = random.sample(scids, 1)[0]

                    qobjs = random.sample(self.cat2obj[qcid], self.num_queries)
                    sobjs = random.sample(self.cat2obj[scid], self.num_shots)
                    for qobj in qobjs:
                        all_samples.append(sobjs + [qobj])

        self.paired_samples = np.array(all_samples)
