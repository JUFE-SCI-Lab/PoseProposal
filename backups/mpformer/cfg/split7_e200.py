from mpformer.cfg.s1 import *

# split6: 60:30
s6_train_cids = [48, 37, 10, 22, 89, 65, 69, 12, 23, 79, 13, 82, 99, 94, 34, 24, 4, 47, 91, 33, 7, 85, 96, 29, 70, 90,
                 76, 68, 38, 81, 35, 63, 88, 73, 100, 95, 71, 15, 87, 54, 27, 43, 30, 55, 16, 20, 28, 42, 83, 44, 3, 39,
                 66, 84, 86, 98, 49, 45, 40, 8]
s6_test_cids = [36, 56, 74, 19, 93, 61, 57, 51, 21, 6, 5, 59, 32, 26, 58, 46, 64, 18, 1, 75, 80, 78, 92, 62, 60, 11,
                97, 67, 25, 72]
# split7: 50:40
s7_train_cids = [48, 37, 10, 22, 89, 65, 69, 12, 23, 79, 13, 82, 99, 94, 34, 24, 4, 47, 91, 33, 7, 85, 96, 29, 70, 90,
                 76, 68, 38, 81,
                 35, 63, 88, 73, 100, 95, 71, 15, 87, 54, 27, 43, 30, 55, 16, 20, 28, 42, 83, 44]
s7_test_cids = [36, 56, 74, 19, 93, 61, 57, 51, 21, 6, 5, 59, 32, 26, 58, 46, 64, 18, 1, 75, 3, 39, 66, 84, 86, 98, 49,
                45, 40, 8, 80, 78, 92, 62, 60, 11, 97, 67, 25, 72]
# split8: 40:50
s8_train_cids = [48, 37, 10, 22, 89, 65, 69, 12, 23, 79, 13, 82, 99, 94, 34, 24, 4, 47, 91, 33, 7, 85, 96, 29, 70, 90,
                 76, 68, 38, 81,
                 35, 63, 88, 73, 100, 95, 71, 15, 87, 54]
s8_test_cids = [36, 56, 74, 19, 93, 61, 57, 51, 21, 6, 5, 59, 32, 26, 58, 46, 64, 18, 1, 75, 27, 43, 30, 55, 16, 20, 28,
                42, 83, 44, 3, 39, 66, 84, 86, 98, 49, 45, 40, 8, 80, 78, 92, 62, 60, 11, 97, 67, 25, 72]

data['train']['ann_file'] = f'{data_root}/annotations/mp100_split5_all.json'
data['val']['ann_file'] = f'{data_root}/annotations/mp100_split5_all.json'
data['test']['ann_file'] = f'{data_root}/annotations/mp100_split5_all.json'
data['viz']['ann_file'] = f'{data_root}/annotations/mp100_split5_all.json'

data['train']['valid_class_ids'] = s7_train_cids
data['val']['valid_class_ids'] = s7_test_cids
data['test']['valid_class_ids'] = s7_test_cids
data['viz']['valid_class_ids'] = s7_test_cids

data['samples_per_gpu'] = 4
data['workers_per_gpu'] = 4

data['test']['num_episodes'] = 10
data['val'] = data['test']

evaluation['interval'] = 10

lr_config['step'] = [160, 180]
total_epochs = 100

checkpoint_config['interval'] = 200
