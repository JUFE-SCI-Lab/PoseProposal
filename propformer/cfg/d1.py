from propformer.cfg.s1 import *

data['workers_per_gpu'] = 0

evaluation['interval'] = 30

lr_config['warmup_iters'] = 1
log_config['interval'] = 1

checkpoint_config['interval'] = 200
total_epochs = 20

data['train']['epoch_sample_num'] = 1000
