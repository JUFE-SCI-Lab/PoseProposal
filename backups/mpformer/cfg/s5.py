from mpformer.cfg.s1 import *

data['train']['ann_file'] = f'{data_root}/annotations/mp100_split5_train.json'
data['val']['ann_file'] = f'{data_root}/annotations/mp100_split5_val.json'
data['test']['ann_file'] = f'{data_root}/annotations/mp100_split5_test.json'
data['viz']['ann_file'] = f'{data_root}/annotations/mp100_split5_test.json'
model['vizcfg']['work_dir'] = f'output/viz_s5'
