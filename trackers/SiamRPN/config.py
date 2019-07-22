# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# Revised for THOR by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

class TrackerConfig(object):
    windowing = 'cosine'
    exemplar_size = 127
    instance_size = 271
    instance_size_glob = 767
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1
    score_size_glob = (instance_size_glob-exemplar_size)/total_stride+1
    context_amount = 0.5
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    anchor_glob = []
    penalty_k = 0.0584502
    window_influence = 0.357794
    lr = 0.353687
    adaptive = True

    def update(self, cfg):
        for k, v in cfg.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1
