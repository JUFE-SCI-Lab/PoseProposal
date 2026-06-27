from propformer.cfg.s1 import *

data['workers_per_gpu'] = 4

evaluation['interval'] = 2
total_epochs = 20

img_split_ratio_train_set = None
img_split_ratio_test_set = None
data['train']['valid_class_ids'] = train_cids_split1
data['val']['valid_class_ids'] = test_cids_split1
data['test']['valid_class_ids'] = test_cids_split1

data['test']['type'] = 'MyTestSet'
data['val']['type'] = data['test']['type']
# CUDA_VISIBLE_DEVICES=0 python train_ours.py --config propformer/cfg/fo_e20.py --work-dir output
# CUDA_VISIBLE_DEVICES=0 python train_ours.py --config propformer/cfg/fo_e20.py --work-dir output --val-only