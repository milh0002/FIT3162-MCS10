import os
import numpy as np
import cv2
import mediapipe as mp
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys
from mediapipe_pose import mediapipe_pose

sys.path.insert(0, '../')


DATA_PATH = "../DATA/"
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

mp = mediapipe_pose()

cTime, pTime = 0, 0

sequence_length = 30
path = "../test/"
actions = os.listdir(path)
for action in actions:
    if not os.path.exists(DATA_PATH + action):
        os.mkdir(DATA_PATH + action)
    video_list = os.listdir(path + action)
    no_sequences = 1776 // len(video_list)
    extra_sequences = 1776 % len(video_list)
    for video in range(len(video_list)):
        cd = os.path.join(path + action + "/" + video_list[video])
        cap = cv2.VideoCapture(cd)
        with mp.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            if video < extra_sequences:
                no_sequences += 1
                pre_sequences = video * no_sequences
            elif video == extra_sequences:
                pre_sequences = video * (no_sequences + 1)
            else:
                pre_sequences = extra_sequences * (no_sequences + 1) + (video - extra_sequences) * no_sequences
            for sequence in range(no_sequences):
                if not os.path.exists(DATA_PATH + action + "/" + str(pre_sequences + sequence)):
                    os.mkdir(DATA_PATH + action + "/" + str(pre_sequences + sequence))
                for frame_num in range(sequence_length):
                    ref, frame = cap.read()
                    try:
                        image, results = mp.mediapipe_detection(frame, holistic)
                    except:
                        break

                    mp.draw_styled_landmarks(image, results)
                    cTime = time.time()
                    fps = 1 / (cTime - pTime)
                    pTime = cTime
                    cv2.putText(image, "FPS:" + str(int(fps)), (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 190), 2,
                                cv2.LINE_AA)

                    if frame_num == 0:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, video), (5, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, video), (5, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    keypoints = mp.extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(pre_sequences + sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
cap.release()
cv2.destroyAllWindows()