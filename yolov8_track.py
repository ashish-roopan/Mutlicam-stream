
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
		camera.out_frame = camera.frame
		camera.print_status()

		print('camera.models', camera.models)

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
		if camera.bboxes.shape[1] == 6:
			camera.bboxes = torch.cat((camera.bboxes[:, :4],  torch.ones((camera.bboxes.shape[0], 1))*-1, camera.bboxes[:, 4:]), dim=1) 	 	
			print('No person tracked')
		
		#. Display
		for x1, y1, x2, y2, obj_id, cls_pred, cls_conf in camera.bboxes:
			x1 = int(x1)
			y1 = int(y1)
			x2 = int(x2)
			y2 = int(y2)

			cv2.rectangle(camera.out_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
			cv2.putText(camera.out_frame, f'{obj_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
			cv2.imshow(cam_id, camera.out_frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				sys.exit(0)
	



	