import sys
import torch
import time
import cv2
import argparse
import time
import numpy as np
from ultralytics import YOLO
from rich import print

sys.path.append('.')
from utils.videoStream import VideoStream
from utils.config_checker import Config_monitor

def draw_results(image, outputs):
	for i, data in enumerate(outputs[0].boxes.data):
		data = data.cpu().numpy()
		x0,y0,x1,y1,score,cls_id = data
		if score > 0 and cls_id == 0:
			cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (255,255,0), 2)
	return image

def tile_frames(streams, num_cams):
	''' Concatenate 4 streams into 1 frame '''
	frames = []
	for i in range(num_cams//4):
		top = np.concatenate([streams[i*4+j]  for j in range(2)], axis=1)
		bottom = np.concatenate([streams[i*4+j]  for j in range(2,4)], axis=1)
		frames.append(np.concatenate([top, bottom], axis=0))
	return frames

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--cfg', default='configs/config.yaml', help='config file')
	parser.add_argument('-d', '--device', default='cuda', help='device')
	return parser.parse_args()

# setup mouse callback to selectr a stream
def mouse_callback(event, x, y, flags, params):
	global row, col, start_flag, exit_flag
	if event == cv2.EVENT_LBUTTONDOWN:
		x = x/1080 * fimg_w
		y = y/1920 * fimg_h

		row = int(y // img_h)
		col = int(x // img_w)
		start_flag = False
		exit_flag = False

def draw_ROIs(frame, ROIs):
	''' Draw ROIs on frame '''
	for ROI in ROIs:
		for pt in ROI:
			cv2.line(frame, tuple(pt), tuple(ROI[(ROI.index(pt)+1)%4]), (0,255,0), 2)

#. Parse args
args = parse_args()
config_monitor = Config_monitor(args.cfg)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_callback)
ROIs = []
ROI = []
start_flag = True
exit_flag = True
k = -1
row = -1
col = -1

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
	img_h, img_w = streams[0].shape[:2]
	print('img_h, img_w', img_h, img_w)


	#.concatenate 4 streams into 1 frame
	frames = tile_frames(streams, num_cams)
	print('tile_frames : ', frames[0].shape)

	#. Run inference
	out_frames = []
	for i, frame in enumerate(frames):
		out_frames.append(frame)

	#. Display
	out_frame = np.concatenate(out_frames, axis=0)
	out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
	fimg_h, fimg_w = out_frame.shape[:2]
	out_frame = cv2.resize(out_frame, (1080, 1920))



	print('row : ', row, 'col : ', col)
	print('start_flag : ', start_flag)
	print('row*2+col : ', row*2+col)
	print()
	#.Display		
	if  start_flag:
		out = out_frame
	else:
		out = streams[row*2+col]
	cv2.imshow('frame', 	)
	k = cv2.waitKey(1)

	print('[bold red] Press space to start [/bold red]')
	while k != 32:
		k = cv2.waitKey(1)
		#.exit
		if k == ord('q'):
			exit()
		
		#.Go back to full view
		if not (start_flag or exit_flag):
			exit_flag = True
			break
			
		if k == 27:
			start_flag = True

		print('start_flag : ', start_flag)
		



	
	

