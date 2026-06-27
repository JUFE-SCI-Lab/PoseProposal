from propformer.cfg.s1 import *

# major
# data['workers_per_gpu'] = 0

evaluation['viz_interval'] = 99999
# model['metacfg']['point_num'] = 100

# Stage-1
data['val']['valid_class_ids'] = train_cids_body
model['test_cfg']['save_meta_pth'] = True
model['test_cfg']['viz_nearest_from'] = None
model['test_cfg']['viz_interval'] = 99999

# Stage-2
data['val']['valid_class_ids'] = test_cids_body
model['test_cfg']['save_meta_pth'] = False
model['test_cfg']['viz_nearest_from'] = f'data/base_metas.pth'
model['test_cfg']['viz_interval'] = 10

# for Quick Debug !
# img_split_ratio_test_set = 0.99
