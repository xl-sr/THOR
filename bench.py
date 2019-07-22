# --------------------------------------------------------
# THOR
# Licensed under The MIT License
# Written by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

import argparse
import numpy as np
from types import SimpleNamespace
import json

from benchmark.vot import test_vot, eval_vot
from benchmark.otb import test_otb, eval_otb
from trackers.tracker import SiamFC_Tracker, SiamRPN_Tracker, SiamMask_Tracker
from benchmark.bench_utils.benchmark_helper import load_dataset

parser = argparse.ArgumentParser(description='Test Trackers on Benchmarks.')
parser.add_argument('-d', '--dataset', dest='dataset', required=True,
                    help='Dataset on which the benchmark is run [VOT2018, OTB2015]')
parser.add_argument('-t', '--tracker', dest='tracker', required=True,
                    help='Name of the tracker [SiamFC, SiamRPN, SiamMask]')
parser.add_argument('--vanilla', action='store_true',
                    help='Run the tracker without THOR')
parser.add_argument('-v', '--viz', action='store_true',
                    help='Show the tracked scene, the stored templated and the modulated view')
parser.add_argument('--verbose', action='store_true',
                    help='Print additional info about THOR')
parser.add_argument('--lb_type', type=str, default='ensemble',
                    help='Specify the type of lower bound [dynamic, ensemble]')
parser.add_argument('--spec_video', type=str, default='',
                    help='Pick a specific video by name, e.g. "lemming" on OTB2015')
parser.add_argument('--save_path', dest='save_path', default='Tracker',
                    help='Name where the tracked trajectory is stored')

def load_cfg(args):
    json_path = f"configs/{args.tracker}/"
    json_path += f"{args.dataset}_"
    if args.vanilla:
        json_path += "vanilla.json"
    else:
        json_path += f"THOR_{args.lb_type}.json"
    cfg = json.load(open(json_path))
    return cfg

def run_bench(delete_after=False):
    args = parser.parse_args()

    cfg = load_cfg(args)
    cfg['THOR']['viz'] = args.viz
    cfg['THOR']['verbose'] = args.verbose

    # setup tracker and dataset
    if args.tracker == 'SiamFC':
        tracker = SiamFC_Tracker(cfg)
    elif args.tracker == 'SiamRPN':
        tracker = SiamRPN_Tracker(cfg)
    elif args.tracker == 'SiamMask':
        tracker = SiamMask_Tracker(cfg)
    else:
        raise ValueError(f"Tracker {args.tracker} does not exist.")

    dataset = load_dataset(args.dataset)
    # optionally filter for a specific videos
    if args.spec_video:
        dataset = {args.spec_video: dataset[args.spec_video]}

    if args.dataset=="VOT2018":
        test_bench, eval_bench = test_vot, eval_vot
    elif args.dataset=="OTB2015":
        test_bench, eval_bench = test_otb, eval_otb
    else:
        raise NotImplementedError(f"Procedure for {args.dataset} does not exist.")

    # testing
    total_lost = 0
    speed_list = []
    for v_id, video in enumerate(dataset.keys(), start=1):
        speed = test_bench(v_id, tracker, dataset[video], args)
        speed_list.append(speed)

    # evaluation
    mean_fps = np.mean(np.array(speed_list))
    bench_res = eval_bench(args.save_path, delete_after)
    bench_res['mean_fps'] = mean_fps
    print(bench_res)

    return bench_res

if __name__ == '__main__':
    run_bench()
