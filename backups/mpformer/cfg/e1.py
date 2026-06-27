from mpformer.cfg.s1 import *

data['test']['num_episodes'] = 5
data['val'] = data['test']

model['archicfg']['enhance_sk_on_q'] = False
model['archicfg']['res_layer_num'] = 1
model['archicfg']['skfeat_layer_num'] = 1
model['metadecodercfg']['in_layer_num'] = 1
model['metadecodercfg']['out_layer_num'] = 1
model['refdecodercfg']['in_layer_num'] = 1
model['refdecodercfg']['out_layer_num'] = 1
model['metacfg']['asm_type'] = 'bipart'