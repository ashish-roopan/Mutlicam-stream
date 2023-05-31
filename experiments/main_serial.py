import sys
import torch
import time
import cv2
import argparse
import time
import numpy as np
import onnxruntime as ort

from utils.videoStream import VideoStream
# from utils.helper import draw_results, parse_config, letterbox
# from utils.helper import parse_config, letterbox
from utils.config_checker import Config_monitor

def draw_results(image, outputs):
	for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
		if score > 0.2 and cls_id == 0: 
			box = np.array([x0,y0,x1,y1])
			box = box.round().astype(np.int32).tolist()
			score = round(float(score),3)
			name = str(score)
			cv2.rectangle(image,box[:2],box[2:],(255,255,0),2)
			cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
	return image

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--cfg', default='configs/config.yaml', help='config file')
	parser.add_argument('-d', '--device', default='cuda', help='device')
	parser.add_argument('-w', '--weights', default='yolov7/yolov7x.onnx', help='weights')
	return parser.parse_args()

args = parse_args()
device = args.device
weights = args.weights
config_monitor = Config_monitor(args.cfg)

#. Parse config
# cam_sources, ROIs, POIs = parse_config(args.cfg)
# video_streams = VideoStream(sources=cam_sources)
# num_cams = len(cam_sources) + 4 - len(cam_sources) % 4

#. ONNX Runtime
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device == 'cuda' else ['CPUExecutionProvider']
session = ort.InferenceSession(weights, providers=providers)
outname = [i.name for i in session.get_outputs()]
inname = [i.name for i in session.get_inputs()]


#. Main loop
while True:
	start = time.time()

	#. Parse config
	if config_monitor.updated:
		print('Config file updated')
		cam_sources, ROIs, POIs = config_monitor.parse_config()
		video_streams = VideoStream(sources=cam_sources)
		num_cams = len(cam_sources) + 4 - len(cam_sources) % 4
	
	streams, statuses = video_streams.read()
	
	#.concatenate 4 streams into 1 frame
	frames = []
	for i in range(num_cams//4):
		top = np.concatenate([streams[i*4+j]  for j in range(2)], axis=0)
		bottom = np.concatenate([streams[i*4+j]  for j in range(2,4)], axis=0)
		frames.append(np.concatenate([top, bottom], axis=1))

	#. Run inference
	for i, frame in enumerate(frames):
		og_frame = frame.copy()
		frame = np.expand_dims(frame, axis=0).transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
		frame = np.ascontiguousarray(frame)
		
		#. Predict
		t1= time.time()
		outputs = session.run(outname,{'images':frame})[0]
		t2 = time.time()

		frame = draw_results(og_frame, outputs)
		cv2.imshow(f'frame{i}', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			sys.exit(0)

	tf = time.time()
	# print(f'FPS: {1/(tf-start)}')
	continue

