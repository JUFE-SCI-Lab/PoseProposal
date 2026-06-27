from propformer.cfg.s1 import *

# Viz Data:
# data['viz']['valid_class_ids'] = [5, 53, 16, 28, 41]  # panda, bird
# python viz_ours.py propformer/cfg/s1_viz.py XXX.pth --cfg-options model.test_cfg.viz_interval=1 img_split_ratio_test_set=0.95


data['workers_per_gpu'] = 4
# data['viz']['valid_class_ids'] = [84, 39]
# data['viz']['valid_class_ids'] = [32, 38, 51]
# data['viz']['valid_class_ids'] = [2, 3, 5, 26, 32, 71, ]
# data['viz']['valid_class_ids'] = test_cids_split1
# data['viz']['valid_class_ids'] = [7]
# data.val.valid_class_ids=[91]
# data['viz']['valid_class_ids'] = None

# data['viz']['valid_class_ids'] = [2, 3, 14, 47, 29, 33, 39, 53, 60]  # for viz vis_factors
data['val']['valid_class_ids'] = [2, 3, 14, 29, 33, 53, 60, 84]  # for viz SOTA
