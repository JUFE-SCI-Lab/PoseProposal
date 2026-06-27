from configs.cmp100.capeformer.one_shots.two_stage_split2_config import *

data['test']['num_episodes'] = 10
data['val'] = data['test']

evaluation['interval'] = 10
evaluation['key_indicator'] = 'mPCK'
checkpoint_config['interval'] = 200
# data['test']['pck_threshold_list'] = [0.05, 0.10, 0.15, 0.2]