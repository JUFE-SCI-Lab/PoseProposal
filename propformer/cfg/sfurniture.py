from propformer.cfg.s1 import *

data['workers_per_gpu'] = 4
evaluation['interval'] = 1
total_epochs = 20

data['train']['valid_class_ids'] = train_cids_furniture
data['val']['valid_class_ids'] = test_cids_furniture
data['test']['valid_class_ids'] = test_cids_furniture
data['viz']['valid_class_ids'] = test_cids_furniture
data['train']['epoch_sample_num'] = 1000

evaluation['viz_interval'] = 10
