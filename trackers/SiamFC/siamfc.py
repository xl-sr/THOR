# --------------------------------------------------------
# THOR
# Licensed under The MIT License
# Written by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

import torch
import numpy as np
from .utils import to_one_indexed, to_zero_indexed, crop_and_resize
from .config import TrackerConfig

def SiamFC_init(im, target_pos, target_sz, cfg):
    state = {}
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    target_pos, target_sz = to_zero_indexed(target_pos, target_sz)

    # set the tracker_config
    p = TrackerConfig()
    p.update(cfg)

    # create hanning window
    p.hann_window = np.outer(np.hanning(p.upscale_sz), np.hanning(p.upscale_sz))
    p.hann_window /= p.hann_window.sum()

    # search scale factors
    p.scale_factors = p.scale_step ** np.linspace(
        -(p.scale_num // 2),
        p.scale_num // 2, p.scale_num)
    p.scale_factors_glob = np.array([0.5, 0.75, 1.0, 1.25, 1.75])

    # exemplar image
    avg_chans = np.mean(im, axis=(0, 1))

    # important params for later use
    context = p.context * np.sum(target_sz)
    p.z_sz = np.sqrt(np.prod(target_sz + context))
    p.x_sz = p.z_sz * p.instance_sz / p.exemplar_sz

    target_pos, target_sz = to_one_indexed(target_pos, target_sz)

    # fill the state dict
    state['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = 1.0
    state['p'] = p
    state['avg_chans'] = avg_chans
    return state

def SiamFC_track(state, im, temp_mem):
    p = state['p']
    avg_chans = state['avg_chans']
    inst_sz = p.instance_sz
    scale_factors = p.scale_factors
    old_pos, old_sz = to_zero_indexed(state['target_pos'], state['target_sz'])
    dev = state['device']

    # get instance images
    ims = [crop_and_resize(
        image=im, center=old_pos, size=p.x_sz * f,
        out_size=inst_sz,
        pad_color=avg_chans) for f in scale_factors]
    ims = np.stack(ims, axis=0)
    ims = torch.from_numpy(ims).to(dev).permute([0, 3, 1, 2]).float()

    # track
    target_pos, target_sz, score, scale = temp_mem.batch_evaluate(ims, old_pos, old_sz, p)

    p.x_sz *= float(scale)
    p.z_sz *= float(scale)

    # return 1-indexed and left-top based bounding box
    target_pos, target_sz = to_one_indexed(target_pos, target_sz)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    state['p'] = p
    state['crop'] = ims[1] # get non scaled image
    return state
