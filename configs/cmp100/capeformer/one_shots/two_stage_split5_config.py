from configs.cmp100.capeformer.one_shots.two_stage_split1_config import *

data['train']['ann_file'] = f'{data_root}/annotations/mp100_split5_train.json'
data['val']['ann_file'] = f'{data_root}/annotations/mp100_split5_val.json'
data['test']['ann_file'] = f'{data_root}/annotations/mp100_split5_test.json'
