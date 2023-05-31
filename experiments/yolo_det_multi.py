import torch	
import cv2
import numpy as np
import os
import time
import argparse
from videoStream import VideoStream

import time
import cv2
import argparse
from ultralytics import YOLO

def draw_results(frame, results):
	bboxes = results[0].boxes.xyxyn.cpu().numpy()  
	classes = results[0].boxes.cls.cpu().numpy()
	confs = results[0].boxes.conf.cpu().numpy()
	h, w, _ = frame.shape
	for bbox, cls, conf in zip(bboxes, classes, confs):
		if int(cls) == 0 and int(conf) < 0.7:
			cv2.rectangle(frame, (int(bbox[0] * w), int(bbox[1] * h)), (int(bbox[2] * w), int(bbox[3] * h)), (0, 0, 255))
	return frame

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--src', type=str, default=0, help='source')  # video, 0 for webcam
	return parser.parse_args()

args = parse_args()
model = YOLO('yolov7x.pt') 
source_dir = args.src
video_sources=[os.path.join(source_dir, video) for video in os.listdir(source_dir)]
video_streams = VideoStream(sources=video_sources)

num_videos = len(video_sources)
print('num_videos:', num_videos)

while True:
	frames, statuses = video_streams.read()
	print('statuses:', statuses)
	start = time.time()

	#. Predict pose
	results = model(frames, show=True, classes=[0])
	continue 

	end = time.time()
	print(f'FPS: {1/(end-start):.2f}')

	#. Display
	#. Resize and concatenate frames
	out1 = np.concatenate((frames[0], frames[1], frames[2], frames[3], frames[4]), axis=1)
	out2 = np.concatenate((frames[5], frames[6], frames[7], frames[8], frames[9]), axis=1)
	out3 = np.concatenate((frames[10], frames[11], frames[12], frames[13], frames[14]), axis=1)
	frame = np.concatenate((out1, out2, out3), axis=0)
	frame = cv2.resize(frame, (int(vid_w*2.5), vid_h*2))
	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break