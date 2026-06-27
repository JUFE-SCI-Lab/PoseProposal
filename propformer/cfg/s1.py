from propformer.cfg.info import *
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=2)
evaluation = dict(interval=10,
                  metric=['PCK', 'NME', 'AUC', 'EPE'],
                  key_indicator='pose_mAP',
                  gpu_collect=True,
                  res_folder='',
                  viz_interval=1e5,
                  save_best=True,
                  )
optimizer = dict(type='Adam',
                 lr=1e-4, )

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=0.001,
                 step=[0.8, 0.9])
total_epochs = 200
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), ])

channel_cfg = dict(num_output_channels=1,
                   dataset_joints=1,
                   dataset_channel=[[0, ], ],
                   inference_channel=[0, ], max_kpt_num=100)
# model settings
model = dict(type='MetaPose',
             pretrained='torchvision://resnet50',
             encoder_config=dict(type='ResNet', depth=50, out_indices=(1, 2, 3,)),
             metacfg=dict(global_dim=256, meta_dim=1024, point_num=100, refine_num=3,
                          vl_type='sigmoid',  # [sigmoid,softmax]
                          ref_feat_factor=0.,
                          use_self_att=False,
                          use_neighbors='ex2',  # [off, zero, ex1, ex2]
                          use_link_defatt=True,
                          use_link_pos=False,
                          init_lr_mul=1.,
                          nhead=8,
                          npoint=4,
                          ),
             hypercfg=dict(l1=1.0, vp=0.1, vn=0.05, mp=0.5, mn=0.5),
             train_cfg=dict(viz_interval=10),
             test_cfg=dict(flip_test=False,
                           post_process='default',
                           shift_heatmap=True,
                           modulate_kernel=11,
                           viz_interval=1,
                           save_meta_pth=False,
                           viz_nearest_from=None,
                           record_linking=False,
                           viz_tracking=None,
                           skip_evaluation=False,
                           )
             )

data_cfg = dict(image_size=[256, 256],
                heatmap_size=[64, 64],
                num_output_channels=channel_cfg['num_output_channels'],
                num_joints=channel_cfg['dataset_joints'],
                dataset_channel=channel_cfg['dataset_channel'],
                inference_channel=channel_cfg['inference_channel'])

train_pipeline = [dict(type='LoadImageFromFile'),
                  dict(type='TopDownGetRandomScaleRotation', rot_factor=15, scale_factor=0.15),
                  dict(type='TopDownAffineFewShot'),
                  dict(type='ToTensor'),
                  # dict(type='ColorAug', jiggle=(0.3, 0.3, 0.3, 0.3), pj=1.0),
                  dict(type='NormalizeTensor',
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
                  dict(type='TopDownGenerateTargetFewShot', sigma=2),
                  dict(type='Collect',
                       keys=['img', 'target', 'target_weight'],
                       meta_keys=[
                           'image_file', 'joints_3d', 'joints_3d_visible',
                           'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs', 'category_id'
                       ]),
                  ]

valid_pipeline = [dict(type='LoadImageFromFile'),
                  dict(type='TopDownAffineFewShot'),
                  dict(type='ToTensor'),
                  dict(type='NormalizeTensor',
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
                  dict(type='TopDownGenerateTargetFewShot', sigma=2),
                  dict(type='Collect',
                       keys=['img', 'target', 'target_weight'],
                       meta_keys=[
                           'image_file', 'joints_3d', 'joints_3d_visible',
                           'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs', 'category_id'
                       ]),
                  ]
test_pipeline = valid_pipeline
img_split_ratio_train_set = None
img_split_ratio_test_set = None
ind_order = True

data_root = '/mnt/data/hongren/mp100'
# unified_ann_file = f'{data_root}/annotations/mp100_all_link_0412.json'
unified_ann_file = f'data/mp100_all_link_0412.json'
unified_img_prefix = f'{data_root}/mp100_images_cc'
data = dict(samples_per_gpu=16, workers_per_gpu=8,
            train=dict(type='MyTrainSet',
                       ann_file=unified_ann_file,
                       img_prefix=unified_img_prefix,
                       data_cfg=data_cfg,
                       valid_class_ids=train_cids_split1,
                       max_kpt_num=channel_cfg['max_kpt_num'],
                       num_shots=2,
                       pipeline=train_pipeline,
                       img_split_ratio=img_split_ratio_train_set,
                       epoch_sample_num=None,
                       ind_order=ind_order
                       ),
            val=dict(type='MyTestSet',
                     ann_file=unified_ann_file,
                     img_prefix=unified_img_prefix,
                     data_cfg=data_cfg,
                     valid_class_ids=test_cids_split1,
                     max_kpt_num=channel_cfg['max_kpt_num'],
                     num_shots=1,
                     num_queries=15,
                     num_episodes=100,
                     pck_threshold_list=[0.05, 0.10, 0.15, 0.2],
                     pipeline=test_pipeline,
                     img_split_ratio=img_split_ratio_test_set,
                     ind_order=ind_order
                     ),
            test=dict(type='MyTestSet',  # TestPoseDataset
                      ann_file=unified_ann_file,
                      img_prefix=unified_img_prefix,
                      data_cfg=data_cfg,
                      valid_class_ids=test_cids_split1,
                      max_kpt_num=channel_cfg['max_kpt_num'],
                      num_shots=1,
                      num_queries=15,
                      num_episodes=100,
                      pck_threshold_list=[0.05, 0.10, 0.15, 0.2],
                      pipeline=test_pipeline,
                      img_split_ratio=img_split_ratio_test_set,
                      ind_order=ind_order
                      ),
            )

shuffle_cfg = dict(interval=1)
