import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ImageManager:
	gsimage = ''

	def __init__(self, path):
		self.path = path

	def trim_image(self):
		pass

	def coordinates_ocr(self):
		pass

	def __inverte(self, imagem, nome):
		imagem = 255 - imagem
		(_, imagem) = cv2.threshold(imagem, 255, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		cv2.imwrite(nome, imagem)

	def apply_mask(self):
		image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
		# gs_imagem = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		self.__inverte(image, 'invertida.png')
		self.gsimage = 'invertida.png'

	def detect_lines(self):
		self.apply_mask()

		# second version of corner detection, without the inner rectangle but with some borders
		image = cv2.imread(self.gsimage)
		edges = cv2.Canny(image, 50, 150, apertureSize=3)
		# Use HoughLinesP to detect lines
		lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 100, minLineLength=100, maxLineGap=15)
		# Draw lines on the image
		for line in lines:
			x1, y1, x2, y2 = line[0]
			cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
		# Display the image
		cv2.imshow('Hough Lines', image)
		cv2.waitKey(0)
