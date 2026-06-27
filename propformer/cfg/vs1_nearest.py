'''
CUDA_VISIBLE_DEVICES=1 python train_ours.py --config propformer/cfg/vs1_nearest.py --val-only --resume-from /data/chenjunjie/pretrained/PoseProposal/s1_e20_N70_g3_vp0.1n0.05_mp0.5n0.5_ld_lr0.0001b4_g104161504_a500/best_pose_mAP_epoch_11.pth
CUDA_VISIBLE_DEVICES=1 python train_ours.py --config propformer/cfg/s1_e20.py --val-only --resume-from /data/chenjunjie/pretrained/PoseProposal/s1_e20_N70_g3_vp0.1n0.05_mp0.5n0.5_ld_lr0.0001b4_g104161504_a500/best_pose_mAP_epoch_11.pth --cfg-options data_cfg.image_size=[512,512]
'''

from propformer.cfg.s1 import *

# major
# data['workers_per_gpu'] = 0

evaluation['viz_interval'] = 99999
# model['metacfg']['point_num'] = 100

# Stage-1
data['val']['valid_class_ids'] = train_cids_split1
model['test_cfg']['save_meta_pth'] = True
model['test_cfg']['viz_nearest_from'] = None
model['test_cfg']['viz_interval'] = 99999

# Stage-2
data['val']['valid_class_ids'] = test_cids_split1
model['test_cfg']['save_meta_pth'] = False
model['test_cfg']['viz_nearest_from'] = f'data/base_metas.pth'
model['test_cfg']['viz_interval'] = 10

data['val']['valid_class_ids'] = [2, 3, 29, 84]

# for Quick Debug !
# img_split_ratio_test_set = 0.99
