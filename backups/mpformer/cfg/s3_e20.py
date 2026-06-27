from mpformer.cfg.s3 import *

data['samples_per_gpu'] = 4
data['workers_per_gpu'] = 4

data['test']['num_episodes'] = 5
data['val'] = data['test']

evaluation['interval'] = 5

total_epochs = 20

checkpoint_config['interval'] = 200