# --------------------------------------------------------
# THOR
# Licensed under The MIT License
# Written by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

import numpy as np
import cv2

def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index

def rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2]), np.array([rect[2], rect[3]])  # 0-index

def crop_and_resize(image, center, size, out_size, pad_color):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - image.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        image = cv2.copyMakeBorder(
            image, npad, npad, npad, npad,
            cv2.BORDER_CONSTANT, value=pad_color)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = image[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size))

    return patch

def to_zero_indexed(pos, sz):
    # convert box to 0-indexed and center based [y, x, h, w]
    box = cxy_wh_2_rect(pos, sz)
    box = np.array([
        box[1] - 1 + (box[3] - 1) / 2,
        box[0] - 1 + (box[2] - 1) / 2,
        box[3], box[2]], dtype=np.float32)
    return box[:2], box[2:]

def to_one_indexed(pos, sz):
    box = np.array([
        pos[1] + 1 - (sz[1] - 1) / 2,
        pos[0] + 1 - (sz[0] - 1) / 2,
        sz[1], sz[0]])
    return rect_2_cxy_wh(box)
