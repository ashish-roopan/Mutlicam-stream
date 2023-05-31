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

def draw_pose(frame, pose, bbox):
	for i in range(17):
		x, y, c = pose[0,i]
		
		xmin, ymin, xmax, ymax = bbox[0,:4]
		if c < 0.7:
			continue    # skip low confidence points
		cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
		cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 255), 2)
	return frame

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source_dir',  help='source dir of videos')  # video, 0 for webcam
	return parser.parse_args()

args = parse_args()
model = YOLO('./yolov8s-pose.pt')  

source_dir = args.source_dir
video_sources=[os.path.join(source_dir, video) for video in os.listdir(source_dir)]
video_sources = video_sources*4
video_streams = VideoStream(sources=video_sources)

cam_ids = range(len(video_sources))
num_videos = len(video_sources)
vid_h, vid_w =  436, 640
out_h = vid_h * 3
out_w = vid_w * 5

frames = np.zeros((num_videos, vid_h, vid_w, 3), dtype=np.uint8)

print('num_videos:', num_videos)

while True:
	streams = video_streams.read()
	start = time.time()

	#. Read frames
	for cam_id in cam_ids:
		try:
			frame = streams[cam_id][1]
			status = streams[cam_id][0]
			if status:
				frames[cam_id] = frame
		except KeyError:
			print(f'cam_id: {cam_id} is not ready')

	read_time = time.time() - start
	print(f'read_fps: {1/read_time:.2f}')

	#. Predict pose
	frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
	print('frames:', frames.shape)
	results = model(frames)
	exit()
	for cam_id in cam_ids:
		frame = frames[cam_id]
		results = model(frame)
		for result in results:
			bbox = result.boxes.data
			pose = result.keypoints
			if len(pose) == 1:
				frame = draw_pose(frame, pose, bbox)
				frames[cam_id] = frame
	
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