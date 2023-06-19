import cv2
import numpy as np
from rich import print
from boxmot import DeepOCSORT
from pathlib import Path


#. A class to represent a camera which will have the following attributes:
#. 	- cam_id
#. 	- source
#. 	- ROI
#. 	- resolution
#. 	- fps
#. 	- status
#. 	- frame
#. 	- thread
#. 	- models['detector', 'tracker', 'gender']
#. 	- centroids history of all people in the frame
#. 	-tracker
#. 	- tracked people


class Camera():
    def __init__(self, cam_id, source, cap, fps, img_h, img_w, ROIs, POIs, models, device):
        self.cam_id = cam_id
        self.source = source
        self.cap = cap
        self.fps = fps
        self.img_h = img_h
        self.img_w = img_w
        self.ROIs = ROIs
        self.POIs = POIs
        self.models = models    
        self.tracker = DeepOCSORT(
            model_weights=Path('osnet_x1_0_dukemtmcreid.pt'),  # which ReID model to use
            device='cuda:0',  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
            fp16=True,  # wether to run the ReID model with half precision or not
                )
        self.tracking = False #. Flag to indicate if the camera cap is used for tracking
        

        self.status_change = False          #Flag to indicate if the status has changed
        self.status = 'ON'
        self.thread = None
        self.restart = False                #Flag to indicate if the thread needs to be restarted
        self.centroid_trace = []
        self.bboxes = None
        self.tracked_people = {}  #. {track_id: [centroids]}
        self.frame = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.frame = cv2.putText(self.frame, f'{self.cam_id} stream loading ...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        

    def print_status(self):
        if self.status_change:
            if self.status == 'ON':
                print(f'[bold #00ff00] {self.cam_id} is {self.status} [/bold #00ff00] \n')
            elif self.status == 'OFF':
                print(f'[bold red] {self.cam_id} is {self.status} [/bold red] ')
                #. Restart the thread
                self.restart = True
            self.status_change = False