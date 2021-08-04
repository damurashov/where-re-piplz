import cv2
from pioneer_sdk import Pioneer
import numpy as np


class PioneerStream:

	def __init__(self):
		self._pioneer = Pioneer()

	def get_image(self):
		camera_frame = None

		while camera_frame is None:
			camera_frame = self._pioneer.get_raw_video_frame()

		camera_frame = cv2.imdecode(np.frombuffer(camera_frame, dtype=np.uint8), cv2.IMREAD_COLOR)

		return camera_frame


if __name__ == "__main__":

	stream = PioneerStream()

	while True:
		frame = stream.get_image()
		cv2.imshow("stream", frame)

		cv2.waitKey(1)
