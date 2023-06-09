import cv2
from boxmot import DeepOCSORT
from pathlib import Path
from ultralytics import YOLO

def draw_results(im, tracker_outputs):
    for x1, y1, x2, y2, track_id, conf, cls in tracker_outputs:
        color = (0, 255, 0)
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(im, str(track_id), (int(x1), int(y1)), 0, 5e-3 * 200, color, 2)
    return im

#.Detector
model = YOLO('yolov8x.pt')

#.Tracker
tracker = DeepOCSORT(
  model_weights=Path('osnet_x0_25_msmt17.pt'),  # which ReID model to use
  device='cuda:0',  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
  fp16=True,  # wether to run the ReID model with half precision or not
)
  
video_path = '/home/ashish/Videos/server_streams/cam2.mp4'
cap = cv2.VideoCapture(video_path)
while True:
    ret, im = cap.read()

    #.Detector
    dets = model(im,verbose=False, classes=[0])[0].boxes.data.cpu().numpy()

    tracker_outputs = tracker.update(dets, im)  

    im = draw_results(im, tracker_outputs)
    cv2.imshow('demo', im)
    if cv2.waitKey(1) == ord('q'):
        break