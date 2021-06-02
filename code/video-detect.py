from imutils.video import FileVideoStream
from imutils import face_utils
import sys
import datetime
import argparse
import math, operator
import imutils
import time
import dlib
import cv2
import numpy as np
from threading import Thread
from queue import Queue
# from main import get_dict
import pandas as pd
import datetime


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", required=True)
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# predictor = dlib.shape_predictor(args["shape_predictor"])

fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)
frame_count = 0
df_pos = 0

prev_diff_array = None
flag =0
fp = open('temp.txt','a')

# final_dict = get_dict()

print("Total frame count:", fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT))
fps = round(fvs.stream.get(cv2.CAP_PROP_FPS))
print("FPS:", fps)



while fvs.more():
    frame = fvs.read()
    curr_time = datetime.datetime.utcfromtimestamp(frame_count/fps)
    print(curr_time)
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    diff_array = 0
    for (i, rect) in enumerate(rects):
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = frame.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
        

        # curr_data = None
        # clone = frame.copy()
        # cv2.putText(clone, "lips", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        # for (x,y) in shape[48:68]:
        #     cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
        #     (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
        #     roi = frame[y:y + h, x:x + w]
        #     roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            
        # roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # res = cv2.resize(roi_gray,(100,250),interpolation = cv2.INTER_CUBIC)
        # curr_data = res
        
    frame_count += 1
    cv2.imshow("Frame ", frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()
fvs.stop()

