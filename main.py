
import sys
import torch
import time
import cv2
import argparse
import time
import numpy as np
from rich import print
from ultralytics import YOLO

from utils.videoStream import VideoStream
from utils.config_checker import Config_monitor


class Hwak:
	def __init__(self, cfg, device, detector_weights):
		self.cfg = cfg
		self.device = device
		self.detector_weights = detector_weights
		self.centroids = {}
		self.num_cams = None
		self.models = None
		self.video_streams = None


		#. Load detector
		self.detector = YOLO(detector_weights)

		#.Start a thread to monitor config file
		self.config_monitor = Config_monitor(cfg)

	def check_for_config_updates(self):
		''' Check for config updates and parse the updated config '''
		if self.config_monitor.updated:
			print('Config file updated')
			cam_sources, ROIs, POIs, models = self.config_monitor.parse_config()
			self.cam_sources = cam_sources
			self.ROIs = ROIs
			self.POIs = POIs
			self.models = models

			self.video_streams = VideoStream(sources=cam_sources)

			self.num_cams = len(cam_sources) + 4 - len(cam_sources) % 4
			self.config_monitor.updated = False
			print('[bold green] Config Parsed [/bold green] ')

	def generate_model_streams(self, streams):
		''' Generate streams for each model '''
		model_streams = {}
		for model, cam_ids in self.models.items():
			print(f'[bold] Generating stream for {model} : {cam_ids} [/bold]')
			model_streams[model] = []
			for cam_id in cam_ids:
				cam_source = f'cam{cam_id}'
				if cam_source in self.cam_sources.keys():
					model_streams[model].append(streams[cam_id])
				
		print('model_streams: ', model_streams.keys)
		return model_streams

	def draw_results(self, image, outputs):
		''' Draw bounding boxes and centroids '''
		cs = []
		for i, data in enumerate(outputs[0].boxes.data):
			data = data.cpu().numpy()
			x0,y0,x1,y1,score,cls_id = data
			if score > 0 and cls_id == 0: 
				cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (255,255,0), 2)
				centroid = (int((x0+x1)/2), int(y1))
				cs.append(centroid)
		return image, cs

	def tile_frames(self, streams, num_cams):
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
	parser.add_argument('-w', '--detector_weights', default='yolov8x.pt', help='detector_weights')
	return parser.parse_args()


args = parse_args()
hwak = Hwak(args.cfg, args.device, args.detector_weights)

#. Main loop
while True:
	start = time.time()

	#. Parse config and check for updates and parse the updated config
	hwak.check_for_config_updates()
	
	#. Read streams
	streams, statuses = hwak.video_streams.read()
	
	#. Generate streams for each model
	models_streams = hwak.generate_model_streams(streams)


	#.concatenate 4 streams into 1 frame
	frames = {}
	for model, streams in models_streams.items():
		frames[model] = hwak.tile_frames(streams, len(streams))

	print('frames: ', frames.keys())
	# frames = hwak.tile_frames(streams, hwak.num_cams)

	#. Detect people
	out_frames = []
	for i, frame in enumerate(frames['det']):
		#. Predict
		t1= time.time()
		results = hwak.detector(frame,verbose=False, classes=[0])
		t2 = time.time()

		frame, cs = hwak.draw_results(frame, results)
		out_frames.append(frame)

		#. Update centroids
		try:
			hwak.centroids[i].append(cs)
		except KeyError:
			hwak.centroids[i] = [cs]

	#. Draw centroids
	for i, cs in hwak.centroids.items():
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
	



