import sys
import cv2
import argparse
import numpy as np
from rich import print
import yaml

sys.path.append('.')
from utils.videoStream import VideoStream
from utils.config_checker import Config_monitor

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--device', default='cuda', help='device')
	parser.add_argument('-v', '--video', default='videos/cam2.mp4', help='video')
	return parser.parse_args()

# setup mouse callback to selectr a stream
def mouse_callback(event, x, y, flags, params):
	global row, col, start_flag, exit_flag
	global ROIs, ROI, frame
	if event == cv2.EVENT_LBUTTONDOWN:
		ROI.append((x,y))
		cv2.circle(frame, (x,y), 2, (0,255,0), -1)
		if len(ROI) == 4:
			print('ROI: ', ROI)
			ROIs.append(ROI)
			ROI = []

def draw_ROIs(frame, ROIs):
	''' Draw ROIs on frame '''
	for ROI in ROIs:
		for pt in ROI:
			cv2.line(frame, tuple(pt), tuple(ROI[(ROI.index(pt)+1)%4]), (0,255,0), 2)
	return frame

#. Parse args
args = parse_args()
cap = cv2.VideoCapture(args.video)
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_callback)

ROI = []
ROIs = []
frame_id = 0


while True:
	#. Read streams
	frame = cap.read()[1]
	print('frame_id: ', frame_id)
	print('ROIs: ', ROIs)

	while True:
		#. Draw ROIs
		if len(ROIs) > 0:
			frame = draw_ROIs(frame, ROIs)
	
		#. Display
		cv2.imshow('frame', frame)
		k = cv2.waitKey(1)
		if k == ord('q'):
			break
		elif k == ord('r'): #. reset
			RIO = []
			ROIs = []

		
	
	





	
	

