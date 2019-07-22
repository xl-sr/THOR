import cv2
import sys, os, glob, re
import json
from os.path import join, dirname, abspath, realpath, isdir
from os import makedirs
import numpy as np
from shutil import rmtree
from ipdb import set_trace
from .bench_utils.bbox_helper import rect_2_cxy_wh, cxy_wh_2_rect

def center_error(rects1, rects2):
    """Center error.
    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2
    centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2
    errors = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=-1))

    return errors

def _intersection(rects1, rects2):
    r"""Rectangle intersection.
    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T

def rect_iou(rects1, rects2, bound=None):
    r"""Intersection over union.
    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape
    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def calc_curves(ious, center_errors, nbins_iou, nbins_ce):
    ious = np.asarray(ious, float)[:, np.newaxis]
    center_errors = np.asarray(center_errors, float)[:, np.newaxis]

    thr_iou = np.linspace(0, 1, nbins_iou)[np.newaxis, :]
    thr_ce = np.arange(0, nbins_ce)[np.newaxis, :]

    bin_iou = np.greater(ious, thr_iou)
    bin_ce = np.less_equal(center_errors, thr_ce)

    succ_curve = np.mean(bin_iou, axis=0)
    prec_curve = np.mean(bin_ce, axis=0)

    return succ_curve, prec_curve

def compute_success_overlap(gt_bb, result_bb):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    iou = overlap_ratio(gt_bb, result_bb)
    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success

def compute_success_error(gt_center, result_center):
    thresholds_error = np.arange(0, 51, 1)
    n_frame = len(gt_center)
    success = np.zeros(len(thresholds_error))
    dist = np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))
    for i in range(len(thresholds_error)):
        success[i] = sum(dist <= thresholds_error[i]) / float(n_frame)
    return success

def get_result_bb(arch, seq):
    result_path = join(arch, seq + '.txt')
    temp = np.loadtxt(result_path, delimiter=',').astype(np.float)
    return np.array(temp)

def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def test_otb(v_id, tracker, video, args):
    toc, regions = 0, []
    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        tic = cv2.getTickCount()
        if f == 0:
            init_pos, init_sz = rect_2_cxy_wh(gt[f])
            state = tracker.setup(im, init_pos, init_sz)
            location = gt[f]
            regions.append(gt[f])
        elif f > 0:
            state = tracker.track(im, state)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(location)
        toc += cv2.getTickCount() - tic

        if args.viz and f > 0:  # visualization
            if f == 0: cv2.destroyAllWindows()
            if len(gt[f]) == 8:
                cv2.polylines(im, [np.array(gt[f], np.int).reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
            else:
                cv2.rectangle(im, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]),
                              (0, 255, 0), 3)
            if len(location) == 8:
                cv2.polylines(im, [location.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]  #
                cv2.rectangle(im, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]),
                              (0, 255, 255), 3)


            cv2.putText(im, "score: {:.4f}".format(state['score']), (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow(video['name'], im)
            cv2.moveWindow(video['name'], 200, 50)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    toc /= cv2.getTickFrequency()

    # save result
    video_path = join('benchmark/results/', args.dataset, args.save_path)
    if not isdir(video_path): makedirs(video_path)
    result_path = join(video_path, '{:s}.txt'.format(video['name']))
    with open(result_path, "w") as fin:
        for x in regions:
            fin.write(','.join([str(i) for i in x])+'\n')

    print('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
           v_id, video['name'], toc, f / toc))
    return f / toc


def eval_otb(save_path, delete_after):
    base_path = join(realpath(dirname(__file__)), '../data', 'OTB2015')
    json_path = base_path + '.json'
    annos = json.load(open(json_path, 'r'))
    seqs = list(annos.keys())

    video_path = join('benchmark/results/OTB2015/', save_path)
    trackers = glob.glob(join(video_path))
    _, _, files = next(os.walk(trackers[0]))
    num_files = len(files)

    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success_overlap = np.zeros((num_files, len(trackers), len(thresholds_overlap)))

    thresholds_error = np.arange(0, 51, 1)
    success_error = np.zeros((num_files, len(trackers), len(thresholds_error)))

    for i, f in enumerate(files):
        seq = f.replace('.txt', '')
        gt_rect = np.array(annos[seq]['gt_rect']).astype(np.float)
        gt_center = convert_bb_to_center(gt_rect)
        for j in range(len(trackers)):
            tracker = trackers[j]
            bb = get_result_bb(tracker, seq)
            center = convert_bb_to_center(bb)
            success_overlap[i][j] = compute_success_overlap(gt_rect, bb)
            success_error[i][j] = compute_success_error(gt_center, center)

    max_auc = 0.0
    max_prec = 0.0
    for i in range(len(trackers)):
        auc = success_overlap[:, i, :].mean()
        if auc > max_auc:
            max_auc = auc

        prec = success_error[:, i, :].mean()
        if prec > max_prec:
            max_prec = prec

    if delete_after:
        rmtree(trackers[0])

    return {'auc': max_auc, 'precision': prec}
