
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
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)

while True:
	start = time.time()

	#. Parse config and check for updates and parse the updated config
	hwak.check_for_config_updates()
	
	#. Read streams
	cameras = hwak.video_streams.read()  #.{cam_id: camera_object}
	
	#. print camera status
	for cam_id, camera in cameras.items():
		camera.print_status()

	for cam_id, camera in cameras.items():
		source = camera.source
		frame = camera.frame
		camera.out_frame = frame.copy()

		#. Detect people  	
		# if 'det' in camera.models:
		# 	camera.bboxes = hwak.detector(frame,verbose=False, classes=[0])[0].boxes.data.cpu()
		# 	camera.bboxes = torch.cat((camera.bboxes[:, :4],  torch.ones((camera.bboxes.shape[0], 1))*-1, camera.bboxes[:, 4:]), dim=1) 	 	

		# #. Detect gender
		# if 'gender' in camera.models:
		# 	camera.bboxes = hwak.gender_detector(frame,verbose=False, classes=[0,1])[0].boxes.data.cpu()
		# 	camera.bboxes = torch.cat((camera.bboxes[:, :4],  torch.ones((camera.bboxes.shape[0], 1))*-1, camera.bboxes[:, 4:]), dim=1) 	 	

		#. Track people		
		if 'track' in camera.models:
			if camera.tracking==False:
				camera.results = camera.detector.track(source=camera.source, show=False, verbose=False, classes=[0], stream=True)
				camera.tracking = True
			
			#. Read the next frame
			result = next(iter(camera.results))
			camera.bboxes = result.boxes.data.cpu() 
			camera.out_frame = result.orig_img

		#. If no person is detected , add a dummy ID 
		if len(camera.bboxes) and camera.bboxes.shape[1] == 6:
			camera.bboxes = torch.cat((camera.bboxes[:, :4],  torch.ones((camera.bboxes.shape[0], 1))*-1, camera.bboxes[:, 4:]), dim=1) 	 	


		# #. Track people		
		# if 'track' in camera.models and camera.tracking==False:
		# 	camera.results = hwak.detector.track(source=source, show=False, verbose=False, classes=[0], stream=True)
		# 	result = next(iter(camera.results))
		# 	boxes = result.boxes.data.cpu() 
		# 	camera.bboxes = boxes
		# 	camera.out_frame = result.orig_img
			
		# 	#. If no person is tracked , add a dummy ID 
		# 	if camera.bboxes is not None and camera.bboxes.shape[1] == 6:
		# 		camera.bboxes = torch.cat((camera.bboxes[:, :4],  torch.ones((camera.bboxes.shape[0], 1))*-1, camera.bboxes[:, 4:]), dim=1) 	 	
		# 	camera.tracking = True


		# elif 'track' in camera.models and camera.tracking==True:
		# 	result = next(iter(camera.results))
		# 	boxes = result.boxes.data.cpu() 
		# 	camera.bboxes = boxes
		# 	camera.out_frame = result.orig_img
		
		# 	#. If no person is tracked , add a dummy ID 
		# 	if camera.bboxes is not None and camera.bboxes.shape[1] == 6:
		# 		camera.bboxes = torch.cat((camera.bboxes[:, :4],  torch.ones((camera.bboxes.shape[0], 1))*-1, camera.bboxes[:, 4:]), dim=1) 	 	




	#. Display
	canvas = hwak.visualize_results(cameras)
	canvas = cv2.putText(canvas, f'FPS: {int(1 / (time.time() - start))}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
	cv2.imshow('Output', canvas)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		sys.exit(0)
	




