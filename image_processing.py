import cv2


RECTANGLE_COLOR = (0, 0, 255,)
RECTANGLE_THICKNESS = 1


def draw_rectangles(image, rectangles):
	"""
	:param image: OpenCV image matrix
	:param rects: Rectangles, tuple of (x,y,w,h) quaternions
	"""

	if rectangles is None:
		return image

	for rect in rectangles:
		image = cv2.rectangle(image, rect[0:2], rect[2:4], RECTANGLE_COLOR, RECTANGLE_THICKNESS)

	return image
