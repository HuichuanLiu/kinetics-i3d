import cv2
import os
import numpy as np
import skvideo.io as sv
from matplotlib import pyplot as pl


def get_optical_flow(file_name, size, max_length=100):
    if not os.path.exists(file_name):
        raise IOError
    # cap = cv2.VideoCapture(file_name)
    # ret, pre_frame = cap.read()
    video = sv.vreader(file_name)
    frm_cnt = 0  # use to record frame number (i.e. length)

    optical_flow = np.zeros((max_length, size, size, 2))
    pre_frame = None

    flipper = 0
    for frame in video:
        if frm_cnt >= max_length:
            break

        # sample 1 frame out of 2
        flipper += 1
        if flipper % 2 != 0:
            continue

        frame = _cropper(frame, 224)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if pre_frame is None:
            pre_frame = frame
            continue

        optical_flow[frm_cnt] = _cal_TVL1(pre_frame, frame)
        frm_cnt += 1
        pre_frame = frame

    optical_flow_rs = _rescale(optical_flow)
    padded_opt_flow = _padder(optical_flow_rs, frm_cnt)

    return padded_opt_flow


def _cropper(frame, size):
    which_edge = np.argmin(frame.shape[0:2])

    if not which_edge:
        width = frame.shape[1] * 1.0 * size / frame.shape[0]
        height = size
    else:
        height = frame.shape[0] * 1.0 * size / frame.shape[1]
        width = size

    frame_resize = cv2.resize(frame, (int(width), int(height)))

    if not which_edge:
        left_bound = (width - size) / 2.0
        right_bound = (width + size / 2.0)
        if right_bound - left_bound != size:
            right_bound += 1
        # return frame_resize[:, int(left_bound):int(right_bound)]
        return frame_resize[:, int(left_bound):int(left_bound) + 224]
    else:
        top_bound = (height - size) / 2.0
        bot_bound = (height + size) / 2.0
        if bot_bound - top_bound != size:
            bot_bound += 1
        # return frame_resize[int(top_bound):int(bot_bound), :]
        return frame_resize[int(top_bound):int(top_bound) + 224, :]


def _cal_TVL1(pframe, frame):
    # alternative method to call DualTVL1 in opencv-contrib-python from pip
    # tvl1 = cv2.DualTVL1OpticalFlow()
    # tvl1 = tvl1.create(0.25, 0.15, 0.3, 5, 5, 0.01, 30, 10, 0.8, 0.0, False)
    tvl1 = cv2.DualTVL1OpticalFlow_create(0.25, 0.15, 0.3, 5, 5, 0.01, 30, 10, 0.8, 0.0, False)
    opt_flow = tvl1.calc(pframe, frame, None)

    return opt_flow


def _padder(opt_flow, frame_num):
    end = opt_flow.shape[0]

    for i in range(frame_num, end):
        opt_flow[i] = opt_flow[(i - frame_num)]

    return opt_flow


def _rescale(opt_flow):
    opt_flow[opt_flow > 20] = 20
    opt_flow[opt_flow < -20] = -20
    opt_flow = opt_flow / 20.
    return opt_flow

# get_optical_flow('/home/g8682/PycharmProjects/kinetics-i3d/data/train/0/6510936896090712631.mp4', 224, 10)
