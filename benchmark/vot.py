import cv2
from os import makedirs
from os.path import join, realpath, dirname, isdir, isfile
import numpy as np
import itertools
from shutil import rmtree

from .bench_utils.pysot.datasets import VOTDataset
from .bench_utils.pysot.evaluation import AccuracyRobustnessBenchmark, EAOBenchmark
from .bench_utils.pyvotkit.region import vot_overlap, vot_float2str
from .bench_utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from .bench_utils.benchmark_helper import load_dataset

def test_vot(v_id, tracker, video, args):
    regions = []  # result and states[1 init / 2 lost / 0 skip]
    image_files, gt = video['image_files'], video['gt']

    start_frame, end_frame, lost_times, toc = 0, len(image_files), 0, 0

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        tic = cv2.getTickCount()
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])

            state = tracker.setup(im, target_pos, target_sz, f)

            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(1)

        elif f > start_frame:  # tracking

            state = tracker.track(im, state)  # track

            if tracker.mask:
                location = state['polygon'].flatten()
                mask = state['mask']
            else:
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                mask = []

            gt_polygon = ((gt[f][0], gt[f][1]), (gt[f][2], gt[f][3]),
                          (gt[f][4], gt[f][5]), (gt[f][6], gt[f][7]))
            if tracker.mask:
                pred_polygon = ((location[0], location[1]), (location[2], location[3]),
                                  (location[4], location[5]), (location[6], location[7]))
            else:
                pred_polygon = ((location[0], location[1]),
                                (location[0] + location[2], location[1]),
                                (location[0] + location[2], location[1] + location[3]),
                                (location[0], location[1] + location[3]))
            b_overlap = vot_overlap(gt_polygon, pred_polygon, (im.shape[1], im.shape[0]))

            if b_overlap:
                regions.append(location)
            else:  # lost
                regions.append(2)
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append(0)
        toc += cv2.getTickCount() - tic

        # visualization (skip lost frame)
        if args.viz and f >= start_frame:
            im_show = im.copy()
            if f == 0: cv2.destroyAllWindows()
            if gt.shape[0] > f:
                if len(gt[f]) == 8:
                    cv2.polylines(im_show, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                else:
                    cv2.rectangle(im_show, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 3)
            if len(location) == 8:
                if tracker.mask:
                    mask = mask > state['p'].seg_thr
                    im_show[:, :, 2] = mask * 255 + (1 - mask) * im_show[:, :, 2]
                location_int = np.int0(location)
                cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]
                cv2.rectangle(im_show, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(im_show, str(lost_times), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(im_show, f"{state['score']:.4f}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(video['name'], im_show)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()

    # save result
    video_path = join('benchmark/results/', args.dataset, args.save_path,
                      'baseline', video['name'])
    if not isdir(video_path): makedirs(video_path)
    result_path = join(video_path, '{:s}_001.txt'.format(video['name']))
    with open(result_path, "w") as fin:
        for x in regions:
            fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
                    fin.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')

    print('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
        v_id, video['name'], toc, f / toc, lost_times))

    return f / toc

def eval_vot(tracker, delete_after=False):
    root = join(realpath(dirname(__file__)), '../data')
    tracker_dir = './benchmark/results/VOT2018'
    dataset = VOTDataset('VOT2018', root)

    dataset.set_tracker(tracker_dir, tracker)
    ar_benchmark = AccuracyRobustnessBenchmark(dataset)
    ret = ar_benchmark.eval(tracker)[tracker]

    # accuracy
    overlaps = list(itertools.chain(*ret['overlaps'].values()))
    acc = np.nanmean(overlaps)

    # robustness
    failures = list(ret['failures'].values())
    length = sum([len(x) for x in ret['overlaps'].values()])
    robustness = np.mean(np.sum(np.array(failures), axis=0) / length) * 100

    # expected average overlap
    eao_benchmark = EAOBenchmark(dataset)
    eao = eao_benchmark.eval(tracker)[tracker]['all']

    res = {'acc': acc, 'robustness': robustness, 'eao': eao}

    if delete_after: # doing param search
        rmtree(join(tracker_dir, tracker))

    return res
