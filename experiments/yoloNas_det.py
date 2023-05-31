import time
import torch
import cv2
import numpy as np
import argparse

from super_gradients.training import models
from super_gradients.common.object_names import Models


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
	parser.add_argument('-s', '--src', type=str, default=0, help='source')  # video, 0 for webcam
	parser.add_argument('-d', '--device', default='cuda', help='device')
	parser.add_argument('-w', '--weights', default='yolov7/yolov7x.onnx', help='weights')
	return parser.parse_args()

args = parse_args()
weights = args.weights

net = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
cap = cv2.VideoCapture(args.src)

while True:
	ret, frame = cap.read()
	if not ret:
		break
	start = time.time()

	t1 = time.time()
	#. Predict 	
	predictions = net.predict(frame)
	# predictions.show()

	t2 = time.time()
	print(f'FPS: {1/(t2-t1):.2f}')
	
	for prediction in  predictions:
		print('prediction', prediction)
		print()
		print('prediction.prediction', prediction.prediction)
		print()
		
		boxes = prediction[:, :4]
		scores = prediction[:, 4]
		classes = prediction[:, 5]
		for box, score, cls in zip(boxes, scores, classes):
			if score < 0.3 or cls != 0:
				continue
			x1, y1, x2, y2 = box
			cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):	
		break