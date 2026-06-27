from propformer.cfg.s1 import *

# Viz Data:
# data['viz']['valid_class_ids'] = [5, 53, 16, 28, 41]  # panda, bird
# python viz_ours.py propformer/cfg/s1_viz.py XXX.pth --cfg-options model.test_cfg.viz_interval=1 img_split_ratio_test_set=0.95


# data['viz']['valid_class_ids'] = [84, 39]
# data['viz']['valid_class_ids'] = [32, 38, 51]
# data['viz']['valid_class_ids'] = [2, 3, 5, 26, 32, 71, ]
# data['viz']['valid_class_ids'] = test_cids_split1
# data['viz']['valid_class_ids'] = [7]
# data.val.valid_class_ids=[91]

# data['val']['valid_class_ids'] = None
data['val']['valid_class_ids'] = train_cids_split1
# data['val']['valid_class_ids'] = test_cids_split1
#
# data['val']['valid_class_ids'] = [44, ]  # [interesting novel cids]
