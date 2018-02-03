import cv2
import os
import numpy as np


def get_optical_flow(file_name, size, max_length=None):
    if not os.path.exists(file_name):
        raise IOError
    file_name='/home/g8682/PycharmProjects/kinetics-i3d/data/6512010272837485791.mp4'
    cap = cv2.VideoCapture(file_name)
    a = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ret, pre_frame = cap.read()
    frame = None
    frm_cnt = 0  # use to record frame number (i.e. length)

    optical_flow = np.zeros((1, max_length, size, size, 2))

    while ret and frm_cnt < max_length:
        ret, frame = cap.read()
        optical_flow[frm_cnt] = cal_TVL1(pre_frame, frame)
        frm_cnt += 1

    for i in range(frm_cnt, max_length):
        frm_cnt += 1
        optical_flow[frm_cnt] = optical_flow[i]


def cropper(frame, size):
    which_edge = np.argmin(frame.shape[0:2])

    if not which_edge:
        height = size
        width = frame.shape[1] * 1.0 * height / size
    else:
        width = size
        height = frame.shape[0] * 1.0 * width / size

    frame_resize = cv2.resize(frame, (height, width))

    if which_edge:
        left_bound = (width - size) / 2.0
        right_bound = (width + size / 2.0)
        if right_bound - left_bound != size:
            right_bound += 1
        return frame_resize[:, left_bound:right_bound]
    else:
        top_bound = (height - size) / 2.0
        bot_bound = (height + size) / 2.0
        if bot_bound - top_bound != size:
            bot_bound += 1
        return frame_resize[top_bound:bot_bound, :]


def cal_TVL1(pframe, frame):
    tvl1 = cv2.cuda.OpticalFlowDual_TVL1.create(0.25, 0.15, 0.3, 5, 5, 0.01, 300, 0.8, 0.0, False)
    opt_flow = tvl1.calc(pframe, frame)
    return opt_flow

get_optical_flow('./data/6512010272837485791.mp4', 224, 101)