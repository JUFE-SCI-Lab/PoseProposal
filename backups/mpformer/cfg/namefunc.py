import datetime
import socket
import torch


def set_outdir_by_exp_name(cfg):
    ostr = cfg.work_dir
    # ostr += f'_M{cfg.model.metacfg.point_num}'

    cmd = cfg.model.comdecodercfg
    ostr += f'_{cmd.type}'
    ostr += f'{int(cmd.self_att)}'
    ostr += f'{int(cmd.iden_pos)}'
    ostr += f'{int(cmd.coor_pos)}'
    ostr += f'{int(cmd.every_ipt_raw)}'

    cmd = cfg.model.metadecodercfg
    ostr += f'm{cmd.in_layer_num}'
    ostr += f'{cmd.out_layer_num}'
    cmd = cfg.model.refdecodercfg
    ostr += f'r{cmd.in_layer_num}'
    ostr += f'{cmd.out_layer_num}'

    ostr += f'{cfg.model.archicfg.enhance_sk_w_meta}'
    ostr += f'{int(cfg.model.comdecodercfg.meta_init)}'

    if cfg.model.archicfg.enhance_sk_on_s:
        ostr += '_es'
    if cfg.model.archicfg.enhance_sk_on_q:
        ostr += '_eq'

    ostr += f'_lr{cfg.optimizer.lr}b{cfg.batch_size}'
    ostr += f"_g{torch.cuda.device_count()}{datetime.datetime.now().strftime('%m%d%H%M')}"
    cfg.work_dir = ostr
    return
