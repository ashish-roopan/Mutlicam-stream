import sys
import torch
import time
import cv2
import argparse
import time
import numpy as np
from ultralytics import YOLO

from utils.videoStream import VideoStream
from utils.config_checker import Config_monitor



def draw_results(image, outputs):
	cs = []
	for i, data in enumerate(outputs[0].boxes.data):
		data = data.cpu().numpy()
		x0,y0,x1,y1,score,cls_id = data
		if score > 0 and cls_id == 0: 
			# cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (255,255,0), 2)
			centroid = (int((x0+x1)/2), int(y1))
			cs.append(centroid)
	return image, cs

def tile_frames(streams, num_cams):
	''' Concatenate 4 streams into 1 frame '''
	frames = []
	for i in range(num_cams//4):
		top = np.concatenate([streams[i*4+j]  for j in range(2)], axis=0)
		bottom = np.concatenate([streams[i*4+j]  for j in range(2,4)], axis=0)
		frames.append(np.concatenate([top, bottom], axis=1))
	return frames

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--cfg', default='configs/config.yaml', help='config file')
	parser.add_argument('-d', '--device', default='cuda', help='device')
	parser.add_argument('-w', '--weights', default='yolov8x.pt', help='weights')
	return parser.parse_args()


args = parse_args()
device = args.device
weights = args.weights
config_monitor = Config_monitor(args.cfg)
centroids = {}

#. Load model
model = YOLO(weights)

#. Main loop
while True:
	start = time.time()

	#. Parse config
	if config_monitor.updated:
		print('Config file updated')
		cam_sources, ROIs, POIs = config_monitor.parse_config()
		video_streams = VideoStream(sources=cam_sources)
		num_cams = len(cam_sources) + 4 - len(cam_sources) % 4
		config_monitor.updated = False
	
	#. Read streams
	streams, statuses = video_streams.read()
	
	#.concatenate 4 streams into 1 frame
	frames = tile_frames(streams, num_cams)

	#. Run inference
	out_frames = []
	for i, frame in enumerate(frames):
		#. Predict
		t1= time.time()
		results = model(frame,verbose=False, classes=[0])
		t2 = time.time()

		frame, cs = draw_results(frame, results)
		out_frames.append(frame)

		#. Update centroids
		try:
			centroids[i].append(cs)
		except KeyError:
			centroids[i] = [cs]

	#. Draw centroids
	for i, cs in centroids.items():
		for c in cs:
			for x,y in c:
				cv2.circle(out_frames[i], (x,y), 2, (255,0,0), -1)


	#. FPS
	print('FPS:', 1 / (time.time() - start))

	#. Display
	out_frame = np.concatenate(out_frames, axis=0)
	out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
	out_frame = cv2.resize(out_frame, (1920, 1080))
	cv2.imshow('frame', out_frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		sys.exit(0)
	


