from threading import Thread
import cv2, time
import numpy as np
from collections import OrderedDict
from rich import print	

from utils.camera import Camera

class VideoStream(object):
	def __init__(self, sources, img_h, img_w, ROIs, POIs, models, detector_weights, device, fps=30):
		''' Create a VideoStream object for each camera in the config file
			- Create a list of camera objects
			- Create a thread for each camera object to read frames from the stream
			- Start the thread
		'''
		
		#. Create a list of camera objects 
		self.cameras = OrderedDict()
		for i, (cam_id, src) in enumerate(sources.items()):
			cam_ROIs = ROIs[cam_id]
			cam_POIs = POIs[cam_id]
			cam_models = models[cam_id] 

			#. Create a VideoCapture object associated with the source
			cap = self.create_capture(src)

			#.create a camera objects
			camera = Camera(cam_id=cam_id, source=src, cap=cap, fps=fps, img_h=img_h, img_w=img_w, ROIs=cam_ROIs, POIs=cam_POIs, models=cam_models, detector_weights=detector_weights, device=device)
			
			#. Create a thread to read frames from the stream
			thread = Thread(target=self.update, args=(i, camera))
			thread.daemon = True
			camera.thread = thread

			#. Append camera to cameras list			
			self.cameras[cam_id] = camera

		print('cameras', self.cameras)

		#. Start all threads
		for cam_id, camera in self.cameras.items():
			camera.thread.start()
			print('[bold yellow] VideoStream: Thread started for camera {} [/bold yellow]'.format(cam_id))
		print('[bold green] VideoStream: All threads started [/bold green]\n')
		time.sleep(1)

	def update(self, i, camera):
		"""Update the camera object with the latest frame from the stream.
			- Read the frame from the stream
			- Resize the frame
			- Convert the frame to RGB
			- Draw a rectangle on the frame to display the camera id
			- Update the camera object
		"""

		while True:
			if camera.restart:
				camera.cap.release()
				camera.cap = self.create_capture(camera.source)
				camera.restart = False
				print('[bold yellow] VideoStream: Thread restarted for camera {} [/bold yellow]'.format(camera.cam_id))

			if camera.cap.isOpened():
				status, frame = camera.cap.read()
				if status:
					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					frame = cv2.resize(frame, (camera.img_w, camera.img_h))
					
					if camera.status == 'OFF':
						camera.status = 'ON'
						camera.status_change = True
				else:
					frame = np.ones((camera.img_h, camera.img_w, 3), dtype=np.uint8)*140
					cv2.putText(frame, f'{camera.cam_id} stream ended', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
					if camera.status == 'ON':
						camera.status = 'OFF'
						camera.status_change = True

				#. Update camera object		
				camera.frame = frame
				time.sleep(1/camera.fps)


	def create_capture(self, source):
		"""Create a VideoCapture object associated with the source."""
		cap = cv2.VideoCapture(source)
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
		cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
		return cap
	
	def read(self):
		return self.cameras
	