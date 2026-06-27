from configs.cmp100.capeformer.one_shots.two_stage_split1_config import *

data['samples_per_gpu'] = 2
data['workers_per_gpu'] = 0

data['test'] = dict(type='CrossClassQSTestPoseDataset',
                    ann_file=f'{data_root}/annotations/mp100_split1_valtest.json',
                    img_prefix=f'{data_root}/mp100_images_cc',
                    data_cfg=data_cfg,
                    valid_class_ids=None,
                    max_kpt_num=channel_cfg['max_kpt_num'],
                    num_shots=1,
                    num_queries=1,
                    num_episodes=5,
                    pck_threshold_list=[0.05, 0.10, 0.15, 0.2],
                    pipeline=test_pipeline)

