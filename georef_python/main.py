# opencv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import isect_segments_bentley_ottmann.poly_point_isect as bot
import os
import sys


def corner_detect(map_path):
        # first version of corner detection, without the inner rectangle

        # # Load image, grayscale, and Otsu's threshold
        # image = cv2.imread(map_path)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # # Find contours
        # cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # Filter for rectangles and squares
        # for c in cnts:
        #     peri = cv2.arcLength(c, True)
        #     approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        #     if len(approx) == 4:
        #         cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
        # # Display the image
        # cv2.imshow('Detected Rectangles and Squares', image)
        # cv2.waitKey(0)

        # second version of corner detection, without the inner rectangle but with some borders
        # image = cv2.imread(map_path)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # # Use HoughLinesP to detect lines
        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        # # Draw lines on the image
        # for line in lines:
        # 	x1, y1, x2, y2 = line[0]
        # 	cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # # Display the image
        # cv2.imshow('Hough Lines', image)
        # cv2.waitKey(0)
        img = cv2.imread(map_path)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
        
        points = []
        for line in lines:
                for x1, y1, x2, y2 in line:
                        points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        print(lines_edges.shape)
        #cv2.imwrite('line_parking.png', lines_edges)

        intersections = bot.isect_segments(points)

        for inter in intersections:
        a, b = inter
        for i in range(3):
                for j in range(3):
                lines_edges[int(b) + i, int(a) + j] = [0, 255, 0]
                
        cv2.imshow('Hough Lines', img)
        cv2.waitKey(0)


if __name__ == '__main__':
        corner_detect('./maps_test/048011.png')
