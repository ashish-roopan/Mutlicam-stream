import time
import torch
import cv2
import numpy as np
import argparse
from ultralytics import YOLO

def draw_results(frame, results):
    bboxes = results[0].boxes.xyxyn.cpu().numpy()  
    classes = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    h, w, _ = frame.shape
    for bbox, cls, conf in zip(bboxes, classes, confs):
        if int(cls) == 0:
            print('Detected person with confidence:', conf)
            cv2.rectangle(frame, (int(bbox[0] * w), int(bbox[1] * h)), (int(bbox[2] * w), int(bbox[3] * h)), (0, 0, 255), 2)
    return frame

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, default=0, help='source')  # video, 0 for webcam
    return parser.parse_args()

args = parse_args()
model = YOLO('yolov8l.pt') 
cap = cv2.VideoCapture(args.src)
fps = []
inference_fps = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()

    #. Predict
    print('frame.shape:', frame.shape)
    results = model(frame, show=True)

    print('FPS:', 1 / (time.time() - start))

    #. Draw results
    frame = draw_results(frame, results)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
