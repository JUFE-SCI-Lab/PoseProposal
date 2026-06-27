from propformer.cfg.s1 import *

data['workers_per_gpu'] = 4
evaluation['interval'] = 1
total_epochs = 20

data['train']['valid_class_ids'] = train_cids_split7
data['val']['valid_class_ids'] = test_cids_split7
data['test']['valid_class_ids'] = test_cids_split7
data['viz']['valid_class_ids'] = test_cids_split7
