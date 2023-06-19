
import sys
import torch
import time
import cv2
import argparse
import time
import numpy as np
from rich import print

from hwak import Hwak


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--cfg', default='configs/config.yaml', help='config file')
	parser.add_argument('-d', '--device', default='cuda', help='device')
	parser.add_argument('-dw', '--detector_weights', default='yolov8n.pt', help='detector_weights')
	parser.add_argument('-gw', '--gender_weights', default='gender_det.pt', help='gender_weights')
	return parser.parse_args()


print('[bold green] ----------------- Starting  ----------------- [/bold green]\n')
args = parse_args()
hwak = Hwak(args.cfg, args.device, args.detector_weights, args.gender_weights)

while True:
	start = time.time()

	#. Parse config and check for updates and parse the updated config
	hwak.check_for_config_updates()
	
	#. Read streams
	cameras = hwak.video_streams.read()  #.{cam_id: camera_object}
	
	for cam_id, camera in cameras.items():
		camera.print_status()
		
		frame = camera.frame
		source = camera.source

		camera.out_frame = frame.copy()

		#. Track people		
		if 'track' in camera.models:
			results = hwak.detector.track(source=source, show=False, verbose=False, classes=[0], stream=True)
			results = next(iter(results))
			boxes = results.boxes.data
			# print(boxes, boxes.shape)
			print(boxes.shape)
			if boxes.shape[1] != 7:
				break
			print()

		#. Display
		for x1, y1, x2, y2, obj_id, cls_pred, cls_conf in boxes:
			x1 = int(x1)
			y1 = int(y1)
			x2 = int(x2)
			y2 = int(y2)

			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
			cv2.imshow("Output", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				sys.exit(0)
	



