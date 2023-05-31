import os
import cv2
import time
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from videoStream import VideoStream

from super_gradients.training import models
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback


def draw_results(frame, results):
	bboxes = results[0].boxes.xyxyn.cpu().numpy()  
	classes = results[0].boxes.cls.cpu().numpy()
	confs = results[0].boxes.conf.cpu().numpy()
	h, w, _ = frame.shape
	for bbox, cls, conf in zip(bboxes, classes, confs):
		if int(cls) == 0 and int(conf) < 0.7:
			cv2.rectangle(frame, (int(bbox[0] * w), int(bbox[1] * h)), (int(bbox[2] * w), int(bbox[3] * h)), (0, 0, 255))
	return frame

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--src', type=str, default=0, help='source')  # video, 0 for webcam
	return parser.parse_args()

args = parse_args()
source_dir = args.src
video_sources=[os.path.join(source_dir, video) for video in os.listdir(source_dir)]
video_streams = VideoStream(sources=video_sources)

model = models.get("yolo_nas_l", pretrained_weights="coco").cuda()
model.eval()

num_videos = len(video_sources)
print('num_videos:', num_videos)

while True:
	frames, statuses = video_streams.read()
	og_frames = frames.copy()
	frames = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames], axis=0)
	frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float().cuda()
	
	print('frames.shape:', frames.shape)
	print('statuses:', statuses)

	t1 = time.time()
	#. Predict 	
	with torch.no_grad():
		raw_predictions = model(frames)
	predictions = PPYoloEPostPredictionCallback(score_threshold=0.1, nms_threshold=0.4, nms_top_k=1000, max_predictions=300)(raw_predictions)
	t2 = time.time()
	print(f'FPS: {1/(t2-t1):.2f}')

	for frame, prediction in zip(og_frames, predictions):
		print('prediction', prediction.shape)
		boxes = prediction[:, :4]
		for box in boxes:
			x1, y1, x2, y2 = box
			cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
	cv2.imshow('frame', frame)
	if cv2.waitKey(49) & 0xFF == ord('q'):	
		break
	print()	
