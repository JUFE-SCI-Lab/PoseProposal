from configs.cmp100.capeformer.one_shots.two_stage_split1_config import *

data['samples_per_gpu'] = 4
data['workers_per_gpu'] = 4

data['test']['num_episodes'] = 10
data['val'] = data['test']

evaluation['interval'] = 5
evaluation['key_indicator'] = 'mPCK'

total_epochs = 60
checkpoint_config['interval'] = 200