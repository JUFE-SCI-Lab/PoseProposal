from configs.cmp100.capeformer.one_shots.two_stage_split1_config import *

data['test']['num_episodes'] = 10
data['val'] = data['test']

evaluation['interval'] = 10
evaluation['key_indicator'] = 'mPCK'
checkpoint_config['interval'] = 200