import cv2
import yaml
import numpy as np


def draw_results(frame, results):
	bboxes = results[0].boxes.xyxyn.cpu().numpy()  
	classes = results[0].boxes.cls.cpu().numpy()
	confs = results[0].boxes.conf.cpu().numpy()
	h, w, _ = frame.shape
	for bbox, cls, conf in zip(bboxes, classes, confs):
		if int(cls) == 0 and int(conf) < 0.7:
			cv2.rectangle(frame, (int(bbox[0] * w), int(bbox[1] * h)), (int(bbox[2] * w), int(bbox[3] * h)), (0, 0, 255))
	return frame

def parse_config(cfg_path):
    with open(cfg_path, 'r') as file:
        cam_config = yaml.safe_load(file)
    
    cameras = cam_config['cameras']
    camera_status = cam_config['camera_status']
    ROIs = cam_config['ROIs']
    POIs = cam_config['POIs']
    
    for cam_id, status in camera_status.items():
        if status != 'active':
            print(cam_id, status)
            del cameras[cam_id]
            del ROIs[cam_id]
            del POIs[cam_id]
    return cameras, ROIs, POIs

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32, r=1, dh=0, dw=0):
    # Resize and pad image while meeting stride-multiple constraints
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)