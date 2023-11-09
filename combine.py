import argparse
import math
import cv2
import numpy as np
import torch
import csv
import os
import cv2
import time
import math
import imutils
import argparse
import numpy as np

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, propagate_ids
from val import normalize, pad_width

# class Pose(object):
#     num_kpts = 18
#     kpt_names = ['nose', 'neck',
#                  'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
#                  'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
#                  'r_eye', 'l_eye',
#                  'r_ear', 'l_ear']

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

nPoints = 18
POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [5, 8], [8, 9], [9,
10], [2, 11], [8, 11], [5, 11], [2, 8], [11, 12], [12, 13], [0, 14], [0, 15], [14, 16],
[15, 17]]
inWidth = 368
inHeight = 368
threshold = 0.1
frame_count = 0
total_time = 0

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def zoom_center(img, zoom_factor):
    y_size = img.shape[0]
    x_size = img.shape[1]
    # define new boundaries
    x1 = int(0.5 * x_size * (1 - 1 / zoom_factor))
    x2 = int(x_size - 0.5 * x_size * (1 - 1 / zoom_factor))
    y1 = int(0.5 * y_size * (1 - 1 / zoom_factor))
    y2 = int(y_size - 0.5 * y_size * (1 - 1 / zoom_factor))
    # first crop image then scale
    img_cropped = img[y1:y2, x1:x2]
    # return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)
    return cv2.resize(img_cropped, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)


def getAngle(a, b, c):
    angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] -
                                                                           b[1], a[0] - b[0]))
    angle = angle + 360 if angle < 0 else angle
    return angle if angle < 180 else 360 - angle

def run_demo(net, height_size, cpu, track_ids):
    frame_count = 0
    total_time = 0
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    keypoints_list = list()

    # green skeleton
    cap_1 = cv2.VideoCapture('./videos/anil.mp4')
    # red skeleton
    # cap_2 = cv2.VideoCapture('./videos/compare2/20230809_183216.mp4')

    # background_image = np.zeros((1920, 1080, 3), np.uint8)
    background_image = np.zeros((960, 540, 3), np.uint8)

    # skip some frames in the beginning of cap_1 and cap_2
    # for i in range(0, 400):
    #     ret_video_1, video_frame_1 = cap_1.read()
    # for i in range(0, 400):
    #     ret_video_2, video_frame_2 = cap_2.read()

    # for img in image_provider:
    ratio_list = list()
    while True:
        t = time.time()
        ret_video_1, video_frame_1 = cap_1.read()
        # ret_video_2, video_frame_2 = cap_2.read()

        if not ret_video_1:
            print('Video has ended')
            break

        # img_1 = video_frame_1.copy()
        # img_2 = video_frame_2.copy()

        # resize image to 540x960
        # frame = cv2.resize(video_frame_1, (540, 960), interpolation=cv2.INTER_CUBIC)
        frame = video_frame_1.copy()

        # orig_video_writer = cv2.VideoWriter('output' + str(int(time.time())) + '.avi',
        #                                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        #                                     10, (frame.shape[1], frame.shape[0]))
        # stick_diag_video_writer = cv2.VideoWriter('stick_diag' + str(int(time.time())) +
        #                                           '.avi',
        #                                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        #                                           10, (frame.shape[1], frame.shape[0]))
        # img_2 = cv2.resize(video_frame_2, (540, 960), interpolation=cv2.INTER_CUBIC)
        stick_diag = np.ones(frame.shape, dtype='uint8') * 0
        img = background_image.copy()
        # import pdb; pdb.set_trace()
        colors = [[0,255,0], [0,0,255]]
        shoulder_distance_list = list()
        # img_list = [img_1, img_2]
        side_len_list = list()
        # for i in [1,2]:
            # transform string img_{i} to img_1 or img_2
            # img = eval('img_' + str(i))
        heatmaps, pafs, scale, pad = infer_fast(net, frame, height_size, stride, upsample_ratio, cpu)
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            if n != 0:
                break
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            keypoints_oneline = list()
            shoulder_distance = 0; left_shoulder = (0, 0); right_shoulder = (0, 0);
            key_points = [(0,0)]*num_keypoints
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                    if kpt_id == 2:
                        left_shoulder = (pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1])
                    if kpt_id == 5:
                        right_shoulder = (pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1])
                    key_points[kpt_id] = (pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1])
                    # keypoints_single = {}
                    # keypoints_single['keypoint_id'] = kpt_id
                    # keypoints_single['keypoint_name'] = Pose.kpt_names[kpt_id]
                    # keypoints_single['keypoint_position'] = (pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1])
                    # keypoints_oneline.append(keypoints_single)
            if left_shoulder != (0, 0) and right_shoulder != (0, 0):
                shoulder_distance = math.sqrt(
                    (left_shoulder[0] - right_shoulder[0]) ** 2 + (left_shoulder[1] - right_shoulder[1]) ** 2)
                shoulder_distance_list.append(shoulder_distance)
            keypoints_list.append(keypoints_oneline)
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
            # pose.draw(background_image, colors[i-1])
            # pose.draw(img_list[i-1], colors[0])

            print("len key_points : ", len(key_points))
            # import pdb; pdb.set_trace()
            if not ((0,0) in [key_points[8], key_points[9], key_points[10]]):
                angle1 = getAngle(key_points[8], key_points[9], key_points[10])
                print("Angle1 : ", angle1)
            else:
                angle1 = -1
            if not ((0,0) in [key_points[13], key_points[12], key_points[11]]):
                angle2 = getAngle(key_points[13], key_points[12], key_points[11])
                print("Angle2 : ", angle2)
            else:
                angle2 = -1

            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]
                ####
                color = (64, 255, 0)
                # if partA in [8, 9, 10] and partB in [8, 9, 10] and 0 < angle1 < 150:
                if partA in [8, 9, 10] and partB in [8, 9, 10] and 0 < angle1 < 90:
                    color = (0, 64, 255)  # change color
                    cv2.putText(frame, "Please correct your right leg angle", (50, 75),
                                cv2.FONT_HERSHEY_COMPLEX, .8, (0, 64, 255), 2,
                                lineType=cv2.LINE_AA)
                # elif partA in [11, 12, 13] and partB in [11, 12, 13] and 0 < angle2 < 150:
                elif partA in [11, 12, 13] and partB in [11, 12, 13] and 0 < angle2 < 90:
                    color = (0, 64, 255)  # color change
                    cv2.putText(frame, "Please correct your left leg angle", (50, 100),
                                cv2.FONT_HERSHEY_COMPLEX, .8, (0, 64, 255), 2,
                                lineType=cv2.LINE_AA)
                else:
                    color = (64, 255, 0)

                if key_points[partA] and key_points[partB]:
                    cv2.line(frame, key_points[partA], key_points[partB], color, 3,
                             lineType=cv2.LINE_AA)
                    cv2.circle(frame, key_points[partA], 8, (0, 0, 255), thickness=-1,
                               lineType=cv2.FILLED)
                    cv2.circle(frame, key_points[partB], 8, (0, 0, 255), thickness=-1,
                               lineType=cv2.FILLED)
                    cv2.line(stick_diag, key_points[partA], key_points[partB], color, 3,
                             lineType=cv2.LINE_AA)
                    cv2.circle(stick_diag, key_points[partA], 8, (125, 125, 125), thickness=-1,
                               lineType=cv2.FILLED)
                    cv2.circle(stick_diag, key_points[partB], 8, (125, 125, 125), thickness=-1,
                               lineType=cv2.FILLED)
            cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2,
                        lineType=cv2.LINE_AA)
            cv2.putText(stick_diag, "time taken = {:.2f} sec".format(time.time() - t), (50,50),
                        cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2,
                        lineType=cv2.LINE_AA)
            cv2.putText(stick_diag, "Right leg angle = {:.2f} degree".format(angle1), (50,80),
                        cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2,
                        lineType=cv2.LINE_AA)
            cv2.putText(stick_diag, "Left leg angle = {:.2f} degree".format(angle2), (50,110),
                        cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2,
                        lineType=cv2.LINE_AA)
            # import pdb; pdb.set_trace()
            cv2.imshow('Output-Skeleton', frame)
            cv2.imshow('canvas', imutils.resize(stick_diag, width=600))
            # orig_video_writer.write(frame)
            # stick_diag_video_writer.write(stick_diag)
            frame_count += 1
            total_time += (time.time() - t)
        # if track_ids == True:
        #     propagate_ids(previous_poses, current_poses)
        #     previous_poses = current_poses
        #     index = 0
        #     for pose in current_poses:
        #         cv2.rectangle(img_list[i-1], (pose.bbox[0], pose.bbox[1]),
        #                       (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        #         cv2.putText(img_list[i-1], 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
        #                     cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        #         if index == 0:
        #             side_len_list.append(pose.bbox[2])
        #         index += 1


        # cv2.waitKey(1)
        key = cv2.waitKey(33)
        if key == 27:  # esc
            return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track-ids', default=True, help='track poses ids')
    args = parser.parse_args()
    #
    # if args.video == '' and args.images == '':
    #     raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    # frame_provider = ImageReader(args.images)
    # # import pdb; pdb.set_trace()
    # if args.video != '':
    #     frame_provider = VideoReader(args.video)

    # list all files in the directory ./videos
    # file_list = os.listdir('./videos')
    # for file_name in file_list:
    #     frame_provider = VideoReader('./videos/' + file_name)
    run_demo(net, args.height_size, args.cpu, args.track_ids)

