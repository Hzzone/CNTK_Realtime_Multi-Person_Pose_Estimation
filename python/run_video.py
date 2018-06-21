import cv2 as cv
import os
import pose_estimation as pe
import imutils
import argparse

#get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='../sample/lucas.m4v')
args = parser.parse_args()

pose = pe.pose_estimation()
cap = cv.VideoCapture(args.video)
active = True

while active:
    _, frame = cap.read()

    if frame is None:
        active = False
        break

    frame = imutils.resize(frame, width=500)
    img = pose.process(frame)
    cv.imshow('feed', img)

    ch = 0xFF & cv.waitKey(1)
    if ch == 27:
        break

cv.destroyAllWindows()
cap.release()
