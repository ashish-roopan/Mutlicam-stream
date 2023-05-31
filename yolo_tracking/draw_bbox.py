# code to draw bounding box on image using mouse callback

import cv2
import numpy as np
import os
import sys
import time
import torch
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='data/video/traffic.mp4', help='path to your video file')
    return parser.parse_args()

#mouse callback function
def draw_ROI(event, x, y, flags, param):
    global ROIs
    global ROI
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        ROI.append([x,y])
        if len(ROI) == 4:
            print('ROI = ', ROI)
            cv2.polylines(img, [np.array(ROI).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            ROI = []


args = parse_args()
ROIs = []
ROI = []
cap = cv2.VideoCapture(args.video)

#set mouse callback function for window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', draw_ROI)

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    og_img = img.copy()
    while len(ROI) < 4:
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            img = og_img.copy()
            ROI = []
    
        for ROI in ROIs:
            if len(ROI) == 4:
                cv2.polylines(img, [np.array(ROI).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
        

