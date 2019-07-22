# --------------------------------------------------------
# THOR
# Licensed under The MIT License
# Written by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

class TrackerConfig(object):
    exemplar_sz = 127
    instance_sz = 255
    context = 0.5
    scale_num = 3
    scale_step = 1.0375
    lr = 0.641662
    penalty_k = 0.982769 
    window_influence = 0.199673
    response_sz = 17
    response_up = 16
    upscale_sz = response_up*response_up
    total_stride = 8
    adjust_scale = 0.001

    def update(self, cfg):
        for k, v in cfg.items():
            if hasattr(self, k):
                setattr(self, k, v)
