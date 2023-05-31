import argparse
import cv2
import os
from typing import List
import numpy as np
import torch

import norfair
from norfair import Detection, Paths, Tracker, Video

from ultralytics import YOLO




def yolo_detections_to_norfair_detections(yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
    ) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=int(detection_as_xywh[-1].item()),
                )
            )
    elif track_points == "bbox":
        # print('yolo_detections.xyxy[0]: ', yolo_detections.xyxy[0])
        detections_as_xyxy = yolo_detections[0].boxes.data
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                )
            )

    return norfair_detections

def parse_args():
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("files", type=str, nargs="+", help="Video files to process")
    parser.add_argument(
        "--detector-path", type=str, default="/yolov7.pt", help="YOLOv7 model path"
    )
    parser.add_argument(
        "--img-size", type=int, default="720", help="YOLOv7 inference size (pixels)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default="0.25",
        help="YOLOv7 object confidence threshold",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default="0.45", help="YOLOv7 IOU threshold for NMS"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=[0],
        help="Filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--device", type=str, default='cuda', help="Inference device: 'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--track-points",
        type=str,
        default="bbox",
        help="Track points: 'centroid' or 'bbox'",
    )
    args = parser.parse_args()
    return args


#. Load model
args = parse_args()
weights = 'yolov8l.pt'
model = YOLO(weights)
video = Video(input_path=args.files[0])


DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000
distance_function = "iou" if args.track_points == "bbox" else "euclidean"
distance_threshold = (
    DISTANCE_THRESHOLD_BBOX
    if args.track_points == "bbox"
    else DISTANCE_THRESHOLD_CENTROID
)
tracker = Tracker(
    distance_function=distance_function,
    distance_threshold=distance_threshold,
)


for frame in video:
    #. Detect bounding boxes
    yolo_detections = model(
        frame,
        conf=args.conf_threshold,
        iou=args.iou_threshold,
        imgsz=args.img_size,
        classes=args.classes,
    )
    detections = yolo_detections_to_norfair_detections(
        yolo_detections, track_points=args.track_points
    )

    #. Update tracker
    tracked_objects = tracker.update(detections=detections)
    
    if args.track_points == "centroid":
        norfair.draw_points(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
    elif args.track_points == "bbox":
        norfair.draw_boxes(frame, detections)
        norfair.draw_tracked_boxes(frame, tracked_objects)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # video.write(frame)