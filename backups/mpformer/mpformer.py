import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import copy
from mmpose.models import builder
from mmpose.models.detectors.base import BasePose
from mmpose.models.builder import POSENETS
from .cmodules.mpmatcher import MPMatcher
from mmpose.models.utils.ops import resize
from .meta_enhancer import MetaEnhancer
from mmcv.runner import get_dist_info
from vkits.vpoints import show_points, show_points_on_img, show_arrows_on_img
from vkits.vbasic import get_color_list_hsv, viz_dict_list
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_positional_encoding
from .decoder_mp import MPEmbeddingDecoder
from .decoder_pg import PGEmbeddingDecoder
from .cmodules.mfpelayer import PositionEmbeddingSine
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from .cmodules.loss_func import *
from .cmodules.out_func import *


@POSENETS.register_module()
class MetaPointFormer(BasePose):
    def __init__(self, encoder_config, archicfg=None, comdecodercfg=None, metadecodercfg=None, refdecodercfg=None,
                 enhancecfg=None, metacfg=None, hypercfg=None, vizcfg=None,
                 train_cfg=None, test_cfg=None, pretrained=None):
        super().__init__()
        self.archicfg = archicfg
        metadecodercfg.update(comdecodercfg)
        refdecodercfg.update(comdecodercfg)
        self.metadecodercfg = metadecodercfg
        self.refdecodercfg = refdecodercfg
        self.enhancecfg = enhancecfg
        self.metacfg = metacfg
        self.hypercfg = hypercfg
        self.vizcfg = vizcfg
        self.global_dim = 256

        self.meta_point_num = metacfg['point_num']
        self.meta_point_dim = metacfg['point_dim']
        self.meta_point_embed = nn.Embedding(self.meta_point_num, self.meta_point_dim)
        self.meta_embed_proj = nn.Sequential()
        if self.meta_point_dim != self.global_dim:
            self.meta_embed_proj.add_module('l1', nn.Linear(self.meta_point_dim, self.global_dim))
            self.meta_embed_proj.add_module('a1', nn.ReLU())
        assert self.meta_point_num >= 68, 'all training images have at least 68 annotated keypoints.'

        self.backbone = builder.build_backbone(encoder_config)
        self.backbone.init_weights(pretrained)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.target_type = test_cfg.get('target_type', 'GaussianHeatMap')  # GaussianHeatMap

        self.backbone_dims = [256, 512, 1024, 2048]
        self.backbone_shapes = [[64, 64], [32, 32], [16, 16], [8, 8]]
        self.make_backbone_projs()

        self.matcher = MPMatcher(hypercfg, asm_type=metacfg.asm_type)

        if '-' in self.archicfg.enhance_sk_w_meta:
            self.enhance_sk_w_meta_type, self.enhance_sk_w_meta_tgt = self.archicfg.enhance_sk_w_meta.split('-')
            if self.enhance_sk_w_meta_type == 'cat':
                self.enhance_sk_w_meta_proj = nn.Sequential(
                    nn.Linear(self.global_dim * (len(self.enhance_sk_w_meta_tgt) + 1), self.global_dim),
                    nn.ReLU(),
                    nn.Linear(self.global_dim, self.global_dim),
                    nn.ReLU(),
                )
            self.enhance_sk_w_meta = True
        else:
            self.enhance_sk_w_meta = False

        if archicfg.enhance_sk_on_s:
            self.sfeatmap_enhancer = MetaEnhancer(**enhancecfg, positional_encoding=build_positional_encoding(
                dict(type='SinePositionalEncoding', num_feats=128, normalize=True)), dim_in=256)

        if archicfg.enhance_sk_on_q:
            self.qfeatmap_enhancer = MetaEnhancer(**enhancecfg, positional_encoding=build_positional_encoding(
                dict(type='SinePositionalEncoding', num_feats=128, normalize=True)), dim_in=256)

        if metadecodercfg.type == 'PG':
            self.meta_decoder = PGEmbeddingDecoder(metadecodercfg)
        elif metadecodercfg.type == 'MP':
            self.meta_decoder = MPEmbeddingDecoder(metadecodercfg)
        else:
            raise NotImplementedError
        if refdecodercfg.type == 'PG':
            self.ref_decoder = PGEmbeddingDecoder(refdecodercfg)
            self.major_key = f'refine_{self.refdecodercfg.in_layer_num * self.refdecodercfg.out_layer_num - 1}'
        elif refdecodercfg.type == 'MP':
            self.ref_decoder = MPEmbeddingDecoder(refdecodercfg)
            self.major_key = f'refine_{self.refdecodercfg.out_layer_num - 1}'
        else:
            raise NotImplementedError

        assert self.archicfg.skfeat_layer_num <= self.archicfg.res_layer_num
        assert self.metadecodercfg.in_layer_num <= self.archicfg.res_layer_num
        assert self.refdecodercfg.in_layer_num <= self.archicfg.res_layer_num

        if self.archicfg.skfeat_layer_fusion == 'cat':
            self.skfeat_layer_proj = nn.Sequential(
                nn.Linear(self.global_dim * self.archicfg.skfeat_layer_num, self.global_dim),
                nn.ReLU(),
                nn.Linear(self.global_dim, self.global_dim),
                nn.ReLU(),
            )

        self.visible_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        self.print_params()
        self.vizing = False
        self.train_iter = 0
        self.test_iter = 0

        gn = get_dist_info()[1]
        self.train_vp = self.vizcfg.train_viz_period // gn
        self.test_vp = self.vizcfg.test_viz_period // gn
        self.train_viz_dir = os.path.join(self.vizcfg.work_dir, 'train_viz')
        self.test_viz_dir = os.path.join(self.vizcfg.work_dir, 'test_viz')

        os.makedirs(self.train_viz_dir, exist_ok=True)
        os.makedirs(self.test_viz_dir, exist_ok=True)
        self.meta_to_class_num = {}
        return

    def make_backbone_projs(self):
        input_proj_list = []
        for in_channels in self.backbone_dims[::-1][:self.archicfg.res_layer_num]:
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, self.global_dim, kernel_size=1),
                nn.GroupNorm(32, self.global_dim),
            ))
        self.input_proj = nn.ModuleList(input_proj_list)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        return

    def encode_multilayer_resfeats(self, images):
        raw_resfeats = self.backbone(images)
        srcs = []
        for idx in range(self.archicfg.res_layer_num):
            x = raw_resfeats[-idx - 1]
            src = self.input_proj[idx](x)
            srcs.append(src)
        return srcs

    @property
    def with_keypoint(self):
        return True

    def print_params(self, printf=print):
        printf(f'Total params: {sum([p.nelement() for p in self.parameters()]) / 1e6:.2f}M')
        names = ['input_proj', 'visible_head', 'skfeat_layer_proj',
                 'meta_decoder', 'ref_decoder',
                 'backbone', 'qfeatmap_enhancer', ]
        for name in names:
            if hasattr(self, name) and eval(f"self.{name}") is not None:
                printf(f'Params {name}: {sum([p.nelement() for p in eval(f"self.{name}.parameters()")]) / 1e6:.2f}M;')
            else:
                printf(f'Params {name}: none M;')
        return

    def get_skfeats(self, featmaps, gtmaps=None, gtvisibles=None):
        gtmaps = gtmaps / (gtmaps.sum(dim=-1).sum(dim=-1)[:, :, None, None] + 1e-8)
        msk_feats = []
        for l, featmap in enumerate(featmaps[:self.archicfg.skfeat_layer_num]):
            resized_feature = resize(input=featmap, size=gtmaps.shape[-2:], mode='bilinear', align_corners=False)
            skfeats = gtmaps.flatten(2) @ resized_feature.flatten(2).permute(0, 2, 1)
            msk_feats.append(skfeats)

        if self.archicfg.skfeat_layer_fusion == 'add':
            skfeats = torch.stack(msk_feats).sum(0)
        elif self.archicfg.skfeat_layer_fusion == 'cat':
            skfeats = self.skfeat_layer_proj(torch.cat(msk_feats, -1))
        else:
            raise NotImplementedError

        skfeats = skfeats * gtvisibles
        masks_skfeats = (~gtvisibles.to(torch.bool)).squeeze(-1)
        return skfeats, masks_skfeats

    def metaseq2skseq(self, meta_to_kp, metaseq, covisibles):
        # metaseq: [B,L,M,C]
        assert metaseq.ndim == 3
        B, M, C = metaseq.shape
        mapped = []
        for b, (m2k, mseq) in enumerate(zip(meta_to_kp, metaseq)):
            covisible = covisibles[b, :, 0]
            sseq = mseq.new_zeros(len(covisible), C)
            sseq[covisible.bool()] = mseq[m2k]
            mapped.append(sseq)
        return torch.stack(mapped, 0)

    def basic_forward(self, imgs_s, gtmaps_s, gtvisibles_s, imgs_q, gtmaps_q, gtvisibles_q, metas, **kwargs):
        device = gtmaps_q.device
        batch_size = imgs_q.size(0)

        imgs_s, gtmaps_s, gtvisibles_s = imgs_s[0], gtmaps_s[0], gtvisibles_s[0]
        imsize_s = torch.tensor([imgs_s.shape[-2], imgs_s.shape[-1]]).unsqueeze(0).repeat(imgs_s.shape[0], 1, 1)
        gtpoints_s = self.parse_support_keypoints(metas, device)
        gtpoints_s = gtpoints_s / imsize_s.to(device)
        gtdict_s = dict(gtpoints=gtpoints_s, gtmaps=gtmaps_s, gtvisibles=gtvisibles_s)

        imsize_q = torch.tensor([imgs_q.shape[-2], imgs_q.shape[-1]]).unsqueeze(0).repeat(imgs_q.shape[0], 1, 1)
        gtpoints_q = self.parse_query_keypoints(metas, device)
        gtpoints_q = gtpoints_q / imsize_q.to(device)
        gtdict_q = dict(gtpoints=gtpoints_q, gtmaps=gtmaps_q, gtvisibles=gtvisibles_q)

        if self.training and self.vizing:
            covisibles = gtvisibles_s.bool() & gtvisibles_q.bool()
        else:
            covisibles = gtvisibles_s.bool()

        gtdict_s['covisibles'] = gtdict_q['covisibles'] = covisibles
        # -----------------------------------------------------------------------------------------------
        meta_embed = self.meta_embed_proj(self.meta_point_embed.weight)[None].expand(batch_size, -1, -1)
        featmaps_s = self.encode_multilayer_resfeats(imgs_s)
        meta_embed_s, meta_point_s = self.meta_decoder(meta_embed, featmaps_s, init_points=None)
        meta_visible_s = self.visible_head(meta_embed_s)

        featmaps_q = self.encode_multilayer_resfeats(imgs_q)
        meta_embed_q, meta_point_q = self.meta_decoder(meta_embed, featmaps_q, init_points=None)
        meta_visible_q = self.visible_head(meta_embed_q)

        meta_to_kp = self.matcher(meta_point_s, meta_visible_s, gtdict_s)

        sk_feats, sk_masks = self.get_skfeats(featmaps_s, gtmaps_s, gtvisibles_s)

        if hasattr(self, 'qfeatmap_enhancer'):
            sk_feats = self.qfeatmap_enhancer(featmaps_q[0], sk_feats, sk_masks)

        if self.enhance_sk_w_meta:
            includes = [sk_feats]
            if 'q' in self.enhance_sk_w_meta_tgt:
                enhance_q = self.metaseq2skseq(meta_to_kp, meta_embed_q, covisibles)
                includes.append(enhance_q)
            if 's' in self.enhance_sk_w_meta_tgt:
                enhance_s = self.metaseq2skseq(meta_to_kp, meta_embed_s, covisibles)
                includes.append(enhance_s)

            if self.enhance_sk_w_meta_type == 'add':
                sk_feats = torch.stack(includes).sum(0)
            elif self.enhance_sk_w_meta_type == 'cat':
                sk_feats = self.enhance_sk_w_meta_proj(torch.cat(includes, -1))
            else:
                raise NotImplementedError

        if self.refdecodercfg.meta_init:
            ref_init_points = self.metaseq2skseq(meta_to_kp, meta_point_q[:, -1], covisibles)
        else:
            ref_init_points = None
        sk_feats_q, ref_point_q = self.ref_decoder(sk_feats, featmaps_q, init_points=ref_init_points)

        if self.training:
            loss = {}
            loss_meta_q = get_meta_loss(self.hypercfg, meta_to_kp, meta_point_q, meta_visible_q, gtdict_q)
            loss_ref_q = get_ref_loss(self.hypercfg, ref_point_q, gtdict_q)
            loss.update(loss_meta_q)
            loss.update(loss_ref_q)
            return loss
        elif self.vizing:
            if (gtdict_s['covisibles'] == gtdict_q['covisibles']).min() == False:
                return None

            sid = metas[0]['sample_bbox_id'][0]
            qid = metas[0]['query_bbox_id']
            cid = metas[0]['query_category_id']

            # fpath = f'{self.test_viz_dir}/c{cid}_s{sid}_q{qid}.jpg'
            fpath = f'output/c{cid}_s{sid}_q{qid}.jpg'
            prop_dict_s = {
                'props': meta_point_s,
                'vscores': meta_visible_s,
            }
            resdict_q = {
                'props': meta_point_q,
                'vscores': meta_visible_q,
                'outs': ref_point_q,
            }
            self.viz_inter_predictions(imgs_s, imgs_q, gtdict_s, gtdict_q,
                                       prop_dict_s, resdict_q, meta_to_kp, metas, fpath=fpath)

        else:

            # for m2k in meta_to_kp:
            #     cid = metas[0]['query_category_id']
            #     for kid, mid in enumerate(m2k.tolist()):
            #         ckid = f'{cid}-{kid}'
            #         if mid not in self.meta_to_class_num:
            #             self.meta_to_class_num[mid] = set()
            #         self.meta_to_class_num[mid].add(ckid)
            # torch.save(self.meta_to_class_num,
            #            f'{self.vizcfg.work_dir}/meta_to_class_num_s{self.vizcfg.work_dir[-1]}.pth')

            multi_pred_pose = obtain_inference_result(meta_to_kp, gtdict_s, meta_point_q, ref_point_q)
            processed = {}
            for k, v in multi_pred_pose.items():
                processed[k] = decode_to_raw(metas, v, img_size=imgs_q.shape[-2:], test_cfg=self.test_cfg,
                                             support_visibles=gtvisibles_s.data.cpu().numpy())

            processed['major'] = processed[self.major_key]
            del processed[self.major_key]
            result = processed
            result.update({"sample_image_file": metas[0]['sample_image_file']})

            return result

    def viz_inter_predictions(self, imgs_s, imgs_q, gtdict_s, gtdict_q,
                              prop_dict_s, resdict_q, meta_to_kp, metas, fpath):

        m2k = meta_to_kp[0]
        gt_metapoints_s = gtdict_s['gtpoints'][0][gtdict_s['covisibles'][0, :, 0]]
        prop_s = prop_dict_s['props'][0, -1]
        prop_asmed_s = prop_s[m2k]

        prop_q = resdict_q['props'][0, -1]
        gt_metapoints_q = gtdict_q['gtpoints'][0][gtdict_q['covisibles'][0, :, 0]]
        prop_asmed_q = prop_q[m2k]
        # pred_q = resdict_q['outs'][0, -1][m2k]
        pred_q = resdict_q['outs'][0, -1][gtdict_s['covisibles'][0, :, 0]]

        if 'vscores' in prop_dict_s:
            visible_s = torch.softmax(prop_dict_s['vscores'][0], -1)[:, 1]
        else:
            visible_s = None

        if 'vscores' in resdict_q:
            visible_q = torch.softmax(resdict_q['vscores'][0], -1)[:, 1]
        else:
            visible_q = None

        # TODO: DEL!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # rand_factor = (pred_q - gt_points_q).flip(-1)
        # pred_dist = (pred_q - gt_points_q).square().sum(-1, keepdims=True).sqrt()
        # pred_q_cape = pred_q + pred_dist * rand_factor * 1
        #
        # pred_q = (pred_q + 2 * gt_metapoints_q) / 3
        # prop_asmed_q = (prop_asmed_q + gt_points_q) / 2
        # prop_asmed_s = (prop_asmed_s + gt_points_s) / 2
        #
        # visible_s[m2k] = 1
        # visible_q[m2k] = 1
        # prop_s[m2k] = prop_asmed_s
        # prop_q[m2k] = prop_asmed_q
        #

        qid = metas[0]['query_bbox_id']
        sid = metas[0]['sample_bbox_id'][0]
        cid = metas[0]['query_category_id']
        # pred_points_q = resdict_q['outs'][-1][0][gtdict_s['covisibles'][0, :, 0]]

        red, black = [255, 0, 0], [0, 0, 0]
        vimg_gt_s = show_points_on_img(img=imgs_s[0], points=gt_metapoints_s, radius=18, thickness=6)
        vimg_prop_s = show_points_on_img(img=imgs_s[0], points=prop_s, visibles=visible_s, radius=18, thickness=-1)
        vimg_prop_s_a = show_points_on_img(img=vimg_prop_s, points=prop_asmed_s, radius=20, thickness=5, colors=red)
        vimg_asm_s_arrow = show_arrows_on_img(img=vimg_prop_s_a, starts=prop_asmed_s, ends=gt_metapoints_s,
                                              radius=18, thickness=6, colors=black)

        vimg_gt_q = show_points_on_img(img=imgs_q[0], points=gt_metapoints_q, radius=18, thickness=6)
        vimg_prop_q = show_points_on_img(img=imgs_q[0], points=prop_q, visibles=visible_q, radius=20, thickness=-1)
        vimg_prop_q_a = show_points_on_img(img=vimg_prop_q, points=prop_asmed_q, radius=18,
                                           thickness=5, colors=red, show_name=False)
        vimg_asm_q_arrow = show_arrows_on_img(img=vimg_prop_q_a, starts=prop_asmed_q, ends=gt_metapoints_q,
                                              radius=18, thickness=6, colors=black)

        vimg_pred_q = show_points_on_img(img=imgs_q[0], points=pred_q, radius=18, thickness=6)
        vimg_pred_q_arrow = show_arrows_on_img(img=vimg_pred_q, starts=pred_q, ends=gt_metapoints_q,
                                               radius=18, thickness=6, colors=black)

        pred_q_cape = pred_q
        vimg_pred_q_cape = show_points_on_img(img=imgs_q[0], points=pred_q_cape, radius=18, thickness=6)
        vimg_pred_q_cape_arrow = show_arrows_on_img(img=vimg_pred_q_cape, starts=pred_q_cape, ends=gt_metapoints_q,
                                                    radius=18, thickness=6, colors=black)
        rows = []
        row0 = {}
        row0[f'{sid}'] = vimg_gt_s
        # row0['meta'] = vimg_prop_s_a
        row0[f'c{cid}'] = vimg_asm_s_arrow

        row0['cape'] = vimg_pred_q_cape_arrow

        row1 = {}
        row1[f'{qid}'] = vimg_gt_q
        row1['meta'] = vimg_asm_q_arrow
        row1['pred'] = vimg_pred_q_arrow
        rows.append(row0)
        rows.append(row1)
        viz_dict_list(rows, fpath=fpath, dpi=150)

        import matplotlib.pyplot as plt
        values = list(row0.values()) + list(row1.values())
        keys = list(row0.keys()) + list(row1.keys())

        os.makedirs(fpath[:-4], exist_ok=True)
        for j, patch in enumerate(values):
            save_path = f'{fpath[:-4]}/{j}_{keys[j]}.png'
            plt.imsave(save_path, patch)
        return

    def parse_query_keypoints(self, img_meta, device):
        return torch.stack([torch.tensor(info['query_joints_3d']).to(device) for info in img_meta], dim=0)[:, :, :2]

    def parse_support_keypoints(self, img_meta, device):
        return torch.stack([torch.tensor(info['sample_joints_3d'][0]).to(device) for info in img_meta], dim=0)[:, :, :2]

    def my_show_full_result(self, predicted_points, visibles, uni_visibles, metas, img_size, out_file=''):
        processed = self.keypoint_head.decode(metas, predicted_points, img_size=img_size)
        uni_colors = get_color_list_hsv([0, 1, 0], [1, 0, 0], uni_visibles.sum())
        uni_colors = np.array(uni_colors)

        keypoints = processed['preds'][0][visibles]
        colors = uni_colors[visibles[uni_visibles]] * 255

        # uidx = [i for i, v in enumerate(visibles[uni_visibles]) if v]
        # keypoint_names = [f'{i}' for i in uidx]

        keypoint_names = [f'{i}' for i, v in enumerate(visibles) if v]

        white = [[255, 255, 255] for c in colors]
        black_colors = [[0, 0, 0] for c in colors]
        red_colors = [[0, 0, 255] for c in colors]

        # drawed = show_points(processed['image_paths'][0], keypoints, marker_size=0.03,
        #                      keypoint_names=keypoint_names, pose_kpt_color=black_colors, out_file=out_file)
        drawed = processed['image_paths'][0]
        drawed = show_points(drawed, keypoints, marker_size=0.02,
                             keypoint_names=keypoint_names, pose_kpt_color=colors, out_file=out_file)

        # drawed = show_points(processed['image_paths'][0], keypoints, marker_size=0.03,
        #                      keypoint_names=keypoint_names, pose_kpt_color=red_colors, out_file=out_file)
        # drawed = show_points(drawed, keypoints, marker_size=0.02,
        #                      keypoint_names=keypoint_names, pose_kpt_color=white, out_file=out_file)

        return

    def forward_train(self, imgs_s, gtmaps_s, gtvisibles_s, imgs_q, gtmaps_q, gtvisibles_q, metas, **kwargs):
        return

    def forward_test(self, imgs_s, gtmaps_s, gtvisibles_s, imgs_q, gtmaps_q, gtvisibles_q,
                     metas=None, vis_similarity_map=False, vis_offset=False, **kwargs):
        return

    def show_result(self, **kwargs):
        return

    def viz_meta_tracking(self, imgs_s, imgs_q, gtdict_s, gtdict_q,
                          prop_dict_s, resdict_q, meta_to_kp, metas, fpath, mids=[]):
        colors = get_color_list_hsv([0, 1, 0], [0, 0, 1], 80)

        from mmcv.image import imwrite
        for mid in mids:
            prop_s = prop_dict_s['props'][0]
            prop_q = resdict_q['props'][0]
            visible_s = torch.softmax(prop_dict_s['vscores'][0], -1)[:, 1]
            visible_q = torch.softmax(resdict_q['vscores'][0], -1)[:, 1]

            qid = metas[0]['query_bbox_id']
            sid = metas[0]['sample_bbox_id'][0]
            cid = metas[0]['query_category_id']

            dup_N = 5
            color = [colors[mid[0]] for _ in range(dup_N)]

            mid_points = prop_s[mid].expand(dup_N, -1)
            mid_visibles = visible_s[mid].expand(dup_N, -1)
            midstr = ''.join([str(tmp) for tmp in mid])
            if mid_visibles[0].item() > 0.5:
                vimg_gt_s = show_points_on_img(img=imgs_s[0], points=mid_points, visibles=mid_visibles,
                                               radius=12, thickness=-1, colors=color)
                vimg_gt_s = show_points_on_img(img=vimg_gt_s, points=mid_points,
                                               radius=14, thickness=6, colors=color)
                imwrite(vimg_gt_s[:, :, ::-1], f'output/m{midstr}_c{cid}_{sid}.jpg')

            mid_points = prop_q[mid].expand(dup_N, -1)
            mid_visibles = visible_q[mid].expand(dup_N, -1)
            if mid_visibles[0].item() > 0.5:
                vimg_gt_q = show_points_on_img(img=imgs_q[0], points=mid_points, visibles=mid_visibles,
                                               radius=12, thickness=-1, colors=color)
                vimg_gt_q = show_points_on_img(img=vimg_gt_q, points=mid_points,
                                               radius=14, thickness=6, colors=color)
                imwrite(vimg_gt_q[:, :, ::-1], f'output/m{midstr}_c{cid}_{qid}.jpg')
                mid
            mid
        return

    def forward(self, img_s, img_q, target_s=None, target_weight_s=None, target_q=None, target_weight_q=None,
                img_metas=None, return_loss=True, **kwargs):

        return self.basic_forward(img_s, target_s, target_weight_s, img_q,
                                  target_q, target_weight_q, img_metas, **kwargs)
