import datetime
import socket
import torch


def set_outdir_by_exp_name(cfg, mode='train'):
    ostr = cfg.work_dir

    if cfg.img_split_ratio_train_set is not None:
        ostr += f'_TrainR{cfg.img_split_ratio_train_set}'
    if cfg.img_split_ratio_test_set is not None:
        ostr += f'_TestR{cfg.img_split_ratio_test_set}'
    cfg.data.train.img_split_ratio = cfg.img_split_ratio_train_set
    cfg.data.val.img_split_ratio = cfg.img_split_ratio_test_set
    cfg.data.test.img_split_ratio = cfg.img_split_ratio_test_set
    # cfg.data.viz.img_split_ratio = cfg.img_split_ratio_test_set

    cfg.data.train.ind_order = cfg.ind_order
    cfg.data.val.ind_order = cfg.ind_order
    cfg.data.test.ind_order = cfg.ind_order
    # cfg.data.viz.ind_order = cfg.ind_order
    ostr += f'_ido' if cfg.ind_order else '_rlo'

    # ostr += f'_N{cfg.model.metacfg.point_num}'
    # ostr += f'_h{cfg.model.metacfg.nhead}p{cfg.model.metacfg.npoint}'

    if cfg.val_only:
        cfg.data.val.data_cfg.image_size = [512, 512]
        # ostr += '_valonly'
        mode = 'val'

    new_steps = []
    for s in cfg.lr_config.step:
        assert 0 < s and s < 1
        new_steps.append(int(cfg.total_epochs * s))
    cfg.lr_config.step = new_steps

    if mode == 'train':
        if cfg.model.type == 'SimpleBaselineModel':
            ostr += f'_hm{cfg.model.hypercfg.l1}'
            ostr += f'_link{cfg.model.hypercfg.mp}'
        elif cfg.model.type == 'GroupPose':
            ostr += f'_l{cfg.model.hypercfg.l1}'
            ostr += f'_vp{cfg.model.hypercfg.vp}n{cfg.model.hypercfg.vn}'
            ostr += f'_link{cfg.model.hypercfg.mp}'
        elif cfg.model.type == 'MetaPoint':
            ostr += f'_point{cfg.model.hypercfg.l1}'
            ostr += f'_visible{cfg.model.hypercfg.vp}n{cfg.model.hypercfg.vn}'
            ostr += f'_link{cfg.model.hypercfg.mp}'
        else:
            ostr += f'_g{cfg.data.train.num_shots + 1}'
            ostr += f'_l{cfg.model.hypercfg.l1}'
            ostr += f'_vp{cfg.model.hypercfg.vp}n{cfg.model.hypercfg.vn}'
            ostr += f'_mp{cfg.model.hypercfg.mp}n{cfg.model.hypercfg.mn}'
            if cfg.model.metacfg.ref_feat_factor != 0.:
                ostr += f'_RefFeatFactor{cfg.model.metacfg.ref_feat_factor}'
            if cfg.model.metacfg.refine_num != 3:
                ostr += f'_RefineNum{cfg.model.metacfg.refine_num}'
            if cfg.model.metacfg.use_self_att:
                ostr += '_sa'
            ostr += f'_n{cfg.model.metacfg.use_neighbors}' if cfg.model.metacfg.use_neighbors != 'ex2' else ''
            if cfg.model.metacfg.use_link_defatt:
                ostr += '_ld'
            if cfg.model.metacfg.use_link_pos:
                ostr += '_lp'

        if cfg.model.metacfg.init_lr_mul != 1.:
            ostr += f'_InitLRMul{cfg.model.metacfg.init_lr_mul}'
        ostr += f'_lr{cfg.optimizer.lr}b{cfg.batch_size}'
        ostr += f'I{cfg.data.train.epoch_sample_num}' if cfg.data.train.epoch_sample_num != None else ''
        ostr += f'E{cfg.total_epochs}' if cfg.total_epochs != 20 else ''

    elif mode == 'test':
        ostr += '_test'
    elif mode == 'val':
        ostr += '_val'
        if cfg.model.test_cfg.viz_tracking is not None:
            ostr += f'_TrackTop{cfg.model.test_cfg.viz_tracking}'
            ostr += f'_ClsNum{len(cfg.data.val.valid_class_ids)}'
    elif mode == 'viz':
        ostr += '_viz'
        cfg.data.viz.data_cfg.image_size = [512, 512]
    else:
        raise NotImplementedError()
    ostr += f"_g{torch.cuda.device_count()}{datetime.datetime.now().strftime('%m%d%H%M')}_{socket.gethostname()[:4]}"
    cfg.work_dir = ostr
    cfg.model.test_cfg.update(dict(viz_dir=f'{ostr}/test_viz'))
    cfg.model.train_cfg.update(dict(viz_dir=f'{ostr}/train_viz'))
    return
