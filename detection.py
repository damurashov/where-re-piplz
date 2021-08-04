"""
The following code is composed using this tutorial https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
as an example
"""

from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
from torch import hub, tensor


class CvDefaultDetector:
	"""
	Wrapper over OpenCV's default detector
	"""

	def __init__(self):
		self.hog = cv2.HOGDescriptor()
		self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	def detect(self, image):
		# detect people in the image
		(rectangles, weights) = self.hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rectangles = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rectangles])
		rectangles = non_max_suppression(rectangles, probs=None, overlapThresh=0.65)

		return rectangles


class HaarCascadeDetector:

	def __init__(self):
		self.detector = cv2.CascadeClassifier('haarcascade_upperbody.xml')

	def detect(self, image):
		# gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return self.detector.detectMultiScale(image, 1.3, 3)


class YoloV5TorchDetector:
	"""
	https://towardsdatascience.com/implementing-real-time-object-detection-system-using-pytorch-and-opencv-70bac41148f7
	"""

	def __init__(self):
		self.model = hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

	def _score_frame(self, image):
		device = 'cpu'
		self.model.to(device)
		# image = [tensor(image)]
		# image = tensor(image)l
		results = self.model(image.real)
		labels = results.xyxyn[0][:, -1].numpy()
		cord = results.xyxyn[0][:, :-1].numpy()

		return labels, cord


	def detect(self, image):
		labels, cords = self._score_frame(image)

		rectangles = []
		for i in range(len(cords)):
			xyxy = [image.shape[1], image.shape[0], image.shape[1], image.shape[0], ]
			cord = [int(cords[i][j] * xyxy[j]) for j in range(4)]

			rectangles += [cord]

		return rectangles  # [(x, y, w, h, score), ...]
