import detection
import imgstream
import image_processing
import cv2


if __name__ == "__main__":
	stream = imgstream.PioneerStream()
	# detector = detection.CvDefaultDetector()
	# detector = detection.HaarCascadeDetector()
	detector = detection.YoloV5Detector()

	while True:
		frame = stream.get_image()
		rectangles = detector.detect(frame)

		print(rectangles)
		frame = image_processing.draw_rectangles(frame, rectangles)

		cv2.imshow("detection", frame)
		cv2.waitKey(1)
