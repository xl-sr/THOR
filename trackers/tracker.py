# --------------------------------------------------------
# THOR
# Licensed under The MIT License
# Written by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

from os.path import dirname, abspath
from ipdb import set_trace
import torch
from trackers.THOR_modules.wrapper import THOR_SiamFC, THOR_SiamRPN, THOR_SiamMask

# SiamFC import
from trackers.SiamFC.net import SiamFC
from trackers.SiamFC.siamfc import SiamFC_init, SiamFC_track

# SiamRPN Imports
from trackers.SiamRPN.net import SiamRPN
from trackers.SiamRPN.siamrpn import SiamRPN_init, SiamRPN_track

# SiamMask Imports
from trackers.SiamMask.net import SiamMaskCustom
from trackers.SiamMask.siammask import SiamMask_init, SiamMask_track
from trackers.SiamMask.utils.load_helper import load_pretrain

class Tracker():
    def __init__(self):
        self.mask = False
        self.temp_mem = None

    def init_func(self, im, pos, sz):
        raise NotImplementedError

    def track_func(self, state, im):
        raise NotImplementedError

    def setup(self, im, target_pos, target_sz, f=0):
        state = self.init_func(im, target_pos, target_sz)
        self.temp_mem.setup(im, target_pos, target_sz, f)
        return state

    def track(self, im, state):
        state = self.track_func(state, im)
        self.temp_mem.update(im, state['crop'], state['target_pos'], state['target_sz'])
        return state

class SiamFC_Tracker(Tracker):
    def __init__(self, cfg):
        super(SiamFC_Tracker, self).__init__()
        self.cfg = cfg

        # setting up the tracker
        model_path = dirname(abspath(__file__)) + '/SiamFC/model.pth'
        model = SiamFC()
        model.load_state_dict(torch.load(model_path))
        self.model = model.eval().cuda()

        # set up template memory
        self.temp_mem = THOR_SiamFC(cfg=cfg['THOR'], net=self.model)

    def init_func(self, im, pos, sz):
        return SiamFC_init(im, pos, sz, self.cfg['tracker'])

    def track_func(self, state, im):
        return SiamFC_track(state, im, self.temp_mem)

class SiamRPN_Tracker(Tracker):
    def __init__(self, cfg):
        super(SiamRPN_Tracker, self).__init__()
        self.cfg = cfg

        # setting up the model
        model_path = dirname(abspath(__file__)) + '/SiamRPN/model.pth'
        model = SiamRPN()
        model.load_state_dict(torch.load(model_path))
        self.model = model.eval().cuda()

        # set up template memory
        self.temp_mem = THOR_SiamRPN(cfg=cfg['THOR'], net=self.model)

    def init_func(self, im, pos, sz):
        return SiamRPN_init(im, pos, sz, self.cfg['tracker'])

    def track_func(self, state, im):
        return SiamRPN_track(state, im, self.temp_mem)

class SiamMask_Tracker(Tracker):
    def __init__(self, cfg):
        super(SiamMask_Tracker, self).__init__()
        self.cfg = cfg
        self.mask = True

        # setting up the model
        model_path = dirname(abspath(__file__)) + '/SiamMask/model.pth'
        model = SiamMaskCustom(anchors=cfg['anchors'])
        model = load_pretrain(model, model_path)
        self.model = model.eval().cuda()

        # set up template memory
        self.temp_mem = THOR_SiamMask(cfg=cfg['THOR'], net=self.model)

    def init_func(self, im, pos, sz):
        return SiamMask_init(im, pos, sz, self.model, self.cfg['tracker'])

    def track_func(self, state, im):
        return SiamMask_track(state, im, self.temp_mem)
