"""
The following code is composed using this tutorial https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
as an example
"""

from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2


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
