from threading import Thread
import cv2, time
import numpy as np
from .helper import letterbox
from rich import print	


class VideoStream(object):
	def __init__(self, sources=[0]):

		#. Add extra cameras to make it divisible by 4
		self.streams = [np.ones((480, 704, 3), dtype=np.uint8)*255 for i in range(len(sources) + 4 - len(sources)%4)]
		self.statuses = {}
		self.fps = 30
		
		#. Create a List of VideoCapture objects
		self.cams = self.init_cams(sources)
		
		#. Create thread for each camera
		self.threads = {}

		for i,(cam_id, cap) in enumerate(self.cams.items()):
			thread = Thread(target=self.update, args=(i, cam_id, cap))
			thread.daemon = True
			self.threads[cam_id] = thread

		#. Start all threads
		for cam_id, thread in self.threads.items():
			thread.start()
		print('[bold yellow] VideoStream: All threads started [/bold yellow] ')
		time.sleep(1)   
		
	def init_cams(self, sources):
		cams = {}
		for cam_id, src in sources.items():
			print(f'[bold green] Reading from camera: {src} [/bold green] ')
			cap = cv2.VideoCapture(src)
			cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
			cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
			cams[cam_id] = cap
		return cams
	
	def update(self, i, cam_id, cap):
		#. Read the next frame from the stream in a different thread
		while True:
			# if cap.isOpened():
			status, frame = cap.read()
			if status:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frame = cv2.resize(frame, (704, 480))
				cv2.rectangle(frame, (0,0), (130,50), (255,255,255), -1)
				cv2.rectangle(frame, (0,0), (130,50), (0,0,0), 2)
				cv2.putText(frame, cam_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
			else:
				frame = np.ones((480, 704, 3), dtype=np.uint8)*140
				print('[bold red] VideoStream: Camera {} is not available [/bold red]'.format(cam_id))
			self.streams[i] = frame
			self.statuses[cam_id] = status
			time.sleep(1/self.fps)
	
	def read(self):
		return self.streams, self.statuses
	
if __name__ == '__main__':
	video_stream_widget = VideoStream()
	while True:
		try:
			video_stream_widget.show_frame()
		except AttributeError:
			pass






# import cv2


# cap = cv2.VideoCapture()
# cap.open("rtsp://admin:Shazabadmin123@172.16.1.8:554")

# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', frame)

#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break