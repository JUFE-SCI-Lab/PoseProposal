import copy
import math
import os
import sys
import cv2
import mmcv
import numpy as np
from mmpose.models import builder
from mmpose.models.detectors.base import BasePose
from mmpose.models.builder import POSENETS
import torch
import torch.nn.functional as F
import torch.nn as nn
from propformer.cpt_modules.predict_head import PredictHead
from .cpt_modules.matcher import Matcher
from .cpt_modules.loss_func import get_pose_loss
from propformer.cmodules.basic_modules import proj_pyramid, make_pyramid_projs, MLP
from mmcv.cnn.bricks.transformer import build_positional_encoding
import time
import json
import subprocess
import time
import webbrowser

@POSENETS.register_module()
class MetaPose(BasePose):
    def __init__(self, encoder_config=None, metacfg=None, hypercfg=None,
                 train_cfg=None, test_cfg=None, pretrained=None):
        super().__init__()
        self.backbone = builder.build_backbone(encoder_config)
        self.backbone.init_weights(pretrained)
        self.global_dim = metacfg['global_dim']
        self.meta_point_num = metacfg['point_num']
        self.refine_num = metacfg['refine_num']
        self.meta_dim = metacfg['meta_dim']
        self.meta_embed = nn.Embedding(self.meta_point_num, self.meta_dim)
        self.meta_embed_proj = nn.Sequential()
        if self.meta_dim != self.global_dim:
            self.meta_embed_proj.add_module('l1', nn.Linear(self.meta_dim, self.global_dim))
            self.meta_embed_proj.add_module('a1', nn.ReLU())
        assert self.meta_point_num >= 68, 'all training images have at least 68 annotated keypoints.'

        self.init_point = nn.Embedding(self.meta_point_num, 2)
        self.init_visible = nn.Embedding(self.meta_point_num, 1)
        self.init_link = nn.Embedding(self.meta_point_num, self.meta_point_num)
        nn.init.constant_(self.init_point.weight, 0.5)
        nn.init.constant_(self.init_visible.weight, 0.5)
        nn.init.constant_(self.init_link.weight, 0.5)

        self.hypercfg = hypercfg
        self.matcher = Matcher(hypercfg)

        self.backbone_projs = make_pyramid_projs([256, 512, 1024, 2048], self.global_dim, num_feature_levels=3)
        self.pos_embed_proj = MLP(self.global_dim, self.global_dim, self.global_dim, 2)
        self.pe_layer = build_positional_encoding(dict(type='SinePositionalEncoding', num_feats=128, normalize=True))

        self.predict_heads = nn.ModuleList()
        for i in range(self.refine_num):
            self.predict_heads.append(PredictHead(
                vl_type=metacfg.vl_type,
                use_self_att=metacfg.use_self_att,
                use_neighbors=metacfg.use_neighbors,
                use_link_defatt=metacfg.use_link_defatt,
                use_link_pos=metacfg.use_link_pos,
                ref_feat_factor=metacfg.ref_feat_factor,
                nhead=metacfg.nhead,
                npoint=metacfg.npoint,
            ))

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg  # {'flip_test': False, 'post_process': 'default', 'shift_heatmap': True, 'modulate_kernel': 11}
        self.target_type = test_cfg.get('target_type', 'GaussianHeatMap')  # GaussianHeatMap

        self.train_viz_interval = train_cfg.viz_interval
        self.train_viz_dir = train_cfg.viz_dir
        os.makedirs(self.train_viz_dir, exist_ok=True)

        self.test_viz_interval = test_cfg.viz_interval
        self.test_viz_dir = test_cfg.viz_dir
        os.makedirs(self.test_viz_dir, exist_ok=True)
        self.test_iter = 0
        self.train_iter = 0
        self.viz_iter = 0
        self.test_cfg = test_cfg
        self.vizing = False

        if test_cfg.viz_nearest_from is None:
            self.did_to_base_metas = {}
        else:
            assert os.path.exists(test_cfg.viz_nearest_from)
            self.did_to_base_metas = torch.load(test_cfg.viz_nearest_from)
            self.base_dids = []
            self.base_points = []
            self.base_visibles = []
            self.base_links = []
            for base_did, base_meta in self.did_to_base_metas.items():
                self.base_dids.append(base_did)
                self.base_points.append(base_meta[1])
                self.base_visibles.append(base_meta[2])
                self.base_links.append(base_meta[3])
            self.base_visibles = torch.stack(self.base_visibles)
            self.base_links = torch.stack(self.base_links)
        if test_cfg.record_linking:
            self.links_sum = 0
        if self.test_cfg.viz_tracking != None:
            # torch.save(torch.rand_like(self.links_sum) * 0.1, 'data/links_sum_noises.pth')
            noise = torch.load('data/links_sum_noises.pth')
            self.links_sum = torch.load(f'data/links_sum.pth').float() + noise
            link_v = torch.triu(self.links_sum, 1).flatten().topk(self.test_cfg.viz_tracking)[0][-1:]
            self.link_vx, self.link_vy = torch.stack(torch.where(self.links_sum == link_v))[:, 0]
            assert self.links_sum[self.link_vx, self.link_vy] == link_v
        return

    @property
    def with_keypoint(self):
        return hasattr(self, 'keypoint_head')

    def forward(self, **kwargs):
        if self.training:
            self.train_iter += 1
            self.test_iter = self.viz_iter = 0
            return self.forward_train(**kwargs)
        elif self.vizing:
            self.viz_iter += 1
            return self.forward_viz(**kwargs)
        else:
            self.test_iter += 1
            return self.forward_test(**kwargs)

    def get_init_pose(self, B, device):
        init_pose = {}
        init_pose['embed'] = self.meta_embed_proj(self.meta_embed.weight)[None].expand(B, -1, -1)

        if hasattr(self, 'init_point'):
            init_pose['point'] = self.init_point.weight[None].expand(B, -1, -1)
            init_pose['visible'] = self.init_visible.weight[None].expand(B, -1, -1).squeeze(2)
            init_pose['link'] = self.init_link.weight[None].expand(B, -1, -1)
        else:
            init_pose['point'] = torch.ones(B, self.meta_point_num, 2).to(device) / 2
            init_pose['visible'] = torch.ones(B, self.meta_point_num).to(device) / 2
            init_pose['link'] = torch.ones(B, self.meta_point_num, self.meta_point_num).to(device) / 2
        return init_pose

    def forward_train(self, **kwargs):
        gt_pose = dict(point=kwargs['points'], visible=kwargs['visibles'], link=kwargs['links'], cid=kwargs['cid'])
        B, G, _, img_H, img_W = kwargs['imgs'].shape
        device = kwargs['imgs'].device
        featpyramid_q = self.backbone(kwargs['imgs'].flatten(0, 1))
        featpyramid_q = proj_pyramid(featpyramid_q, self.backbone_projs)
        last_pose = self.get_init_pose(B * G, device)
        pred_pose_list = []
        for i in range(self.refine_num):
            pred_pose_q = self.predict_heads[i](featpyramid_q, last_pose, self.pos_embed_proj, self.pe_layer)
            last_pose = pred_pose_q
            pred_pose_list.append(pred_pose_q)
        meta_to_kp = self.matcher(pred_pose_list, gt_pose)
        loss_dict = get_pose_loss(self.hypercfg, meta_to_kp, pred_pose_list, gt_pose)
        return loss_dict

    # poesimage
    def forward_test(self, **kwargs):
        gt_pose = dict(point=kwargs['point_q'], visible=kwargs['visible_q'], link=kwargs['link_q'])
        gt_cid = kwargs['cid']
        image = kwargs['img_q']

        highreso_img = image.clone()
        if image.size(-1) != 256:
            image = F.upsample_bilinear(highreso_img, 256)
        B, _, img_H, img_W = image.shape  # B应该是1
        device = image.device
        featpyramid_q = self.backbone(image)
        featpyramid_q = proj_pyramid(featpyramid_q, self.backbone_projs)
        last_pose = self.get_init_pose(B, device)
        pred_pose_list = []

        for i in range(self.refine_num):
            pred_pose_q = self.predict_heads[i](featpyramid_q, last_pose, self.pos_embed_proj, self.pe_layer)
            last_pose = pred_pose_q
            pred_pose_list.append(pred_pose_q)

        return pred_pose_list, gt_pose, kwargs['cid']  # kwargs['dataset_seq_id']


    def show_result(self):
        return