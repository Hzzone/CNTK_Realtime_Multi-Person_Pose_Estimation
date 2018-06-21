import pose_estimation as pe
import os
import cv2 as cv
import argparse

#get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='../sample/ski.jpg')
args = parser.parse_args()

pose = pe.pose_estimation()
oriImg = cv.imread(args.image)
preview = pose.process(oriImg)
cv.imshow('feed', preview)
cv.waitKey(0)