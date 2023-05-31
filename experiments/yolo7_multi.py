import sys
import torch
import time
import cv2
import argparse
import time
import numpy as np
import onnxruntime as ort

from utils.videoStream import VideoStream
from utils.helper import draw_results, parse_config, letterbox


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--cfg', default='configs/config.yaml', help='config file')
	parser.add_argument('-d', '--device', default='cuda', help='device')
	parser.add_argument('-w', '--weights', default='yolov7/yolov7-tiny.onnx', help='weights')
	return parser.parse_args()

args = parse_args()
device = args.device
weights = args.weights

#. Parse config
cam_sources, ROIs, POIs = parse_config(args.cfg)
video_streams = VideoStream(sources=cam_sources)

#. ONNX Runtime
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device == 'cuda' else ['CPUExecutionProvider']
session = ort.InferenceSession(weights, providers=providers)
outname = [i.name for i in session.get_outputs()]
inname = [i.name for i in session.get_inputs()]


#. Main loop
while True:
	frames, statuses = video_streams.read()
	og_frames = frames.copy()
	frames = np.concatenate([np.expand_dims(frame, axis=0) for cam_id, frame in frames], axis=0).transpose((0, 3, 1, 2))
	frames = frames.astype(np.float32) / 255.0
	print('statuses:', statuses)
	print('frames.shape:', frames.shape)
	
	#. Predict
	t1= time.time()
	outputs = session.run(outname,{'images':frames})[0]
	t2 = time.time()
	print('time:', t2-t1, 'FPS:', 1/(t2-t1))
	print('outputs.shape:', outputs.shape)

	#. Draw results
	for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
		if batch_id >= 6:
			break
		if score < 0.5 or cls_id != 0: 
			continue
		image = og_frames[f'cam{int(batch_id)+1}'].copy()
		box = np.array([x0,y0,x1,y1])
		box = box.round().astype(np.int32).tolist()
		score = round(float(score),3)
		name = str(score)
		color = np.random.randint(0,255,(3)).tolist()
		cv2.rectangle(image,box[:2],box[2:],color,2)
		cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
	
		cv2.imshow(f'cam{int(batch_id)+1}',image)
		if cv2.waitKey(1) & 0xFF == ord('q'):	
			break
		print()
	print()













for img in imgList:
  image, ratio, dwdh = letterbox(image, auto=False)
  image = image.transpose((2, 0, 1))
  image = np.expand_dims(image, 0)
  image = np.ascontiguousarray(image)
  im = image.astype(np.float32)
  resize_data.append((im,ratio,dwdh))

np_batch = np.concatenate([data[0] for data in resize_data])
print('np_batch.shape',np_batch.shape)

im = np.ascontiguousarray(np_batch/255)
out = session.run(outname,{'images':im})[0]
print('out',out[0].shape)

for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(out):
	if batch_id >= 6:
		break
	if score < 0.5 or cls_id != 0: 
		continue
	image = origin_RGB[int(batch_id)]
	ratio,dwdh = resize_data[int(batch_id)][1:]
	box = np.array([x0,y0,x1,y1])
	box -= np.array(dwdh*2)
	box /= ratio
	box = box.round().astype(np.int32).tolist()
	score = round(float(score),3)
	name = str(score)
	color = [255,255,0]
	cv2.rectangle(image,box[:2],box[2:],color,2)
	cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
	cv2.imshow(f'image_{i}',image)
cv2.waitKey(0)



