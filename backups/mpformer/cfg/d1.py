from mpformer.cfg.s1 import *

data['samples_per_gpu'] = 2
data['workers_per_gpu'] = 0

data['test']['num_episodes'] = 1
data['val'] = data['test']

evaluation['interval'] = 1

checkpoint_config['interval'] = 200
total_epochs = 60

lr_config['warmup_iters'] = 1
log_config['interval'] = 5
