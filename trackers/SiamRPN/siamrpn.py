# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# Revised for THOR by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from ipdb import set_trace
from .utils import get_subwindow_tracking, generate_anchor
from .config import TrackerConfig

def SiamRPN_init(im, target_pos, target_sz, cfg):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    # set the tracker_config
    p = TrackerConfig()
    p.update(cfg)

    if p.adaptive:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287
        else:
            p.instance_size = 271
        p.score_size = (p.instance_size - p.exemplar_size) // p.total_stride + 1

    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))
    avg_chans = np.mean(im, axis=(0, 1))

    window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state['p'] = p
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = 1.0
    return state

def SiamRPN_track(state, im, temp_mem):
    p = state['p']
    avg_chans = state['avg_chans']
    window = state['window']
    old_pos = state['target_pos']
    old_sz = state['target_sz']
    dev = state['device']

    wc_z = old_sz[1] + p.context_amount * sum(old_sz)
    hc_z = old_sz[0] + p.context_amount * sum(old_sz)

    s_z = round(np.sqrt(wc_z * hc_z))
    s_z = min(s_z, min(state['im_h'], state['im_w']))
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = round(s_z + 2 * pad)

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, old_pos, p.instance_size,\
                                             round(s_x), avg_chans).unsqueeze(0))

    # track
    target_pos, target_sz, score = temp_mem.batch_evaluate(x_crop.to(dev), old_pos, old_sz * scale_z, window, scale_z, p)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    state['crop'] = x_crop
    return state
