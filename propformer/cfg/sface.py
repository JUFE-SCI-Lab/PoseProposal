from propformer.cfg.s1 import *

data['workers_per_gpu'] = 4
evaluation['interval'] = 1
total_epochs = 20

data['train']['valid_class_ids'] = train_cids_face
data['val']['valid_class_ids'] = test_cids_face
data['test']['valid_class_ids'] = test_cids_face
data['viz']['valid_class_ids'] = test_cids_face
data['train']['epoch_sample_num'] = 1000
