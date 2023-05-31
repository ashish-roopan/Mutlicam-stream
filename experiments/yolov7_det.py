import time
import torch
import cv2
import numpy as np
import argparse
import onnxruntime as ort



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

#. ONNX Runtime
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device == 'cuda' else ['CPUExecutionProvider']
session = ort.InferenceSession(weights, providers=providers)
outname = [i.name for i in session.get_outputs()]
inname = [i.name for i in session.get_inputs()]

cap = cv2.VideoCapture(args.src)
fps = []
inference_fps = []

while True:
	ret, frame = cap.read()
	if not ret:
		break
	start = time.time()

	#. Preprocess
	frame = cv2.resize(frame, (640, 640))
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	og_frame = frame.copy()
	frame = np.expand_dims(frame, axis=0).transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
	frame = np.ascontiguousarray(frame)

	#. Predict
	outputs = session.run(outname,{'images':frame})[0]


	end = time.time()
	print(f'FPS: {1/(end-start):.2f}')

	#. Draw results
	frame = draw_results(og_frame, outputs)
	cv2.imshow('frame', frame)
	if cv2.waitKey(49) & 0xFF == ord('q'):
		break
