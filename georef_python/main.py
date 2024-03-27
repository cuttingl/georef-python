#opencv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys


def corner_detect(map_path):
    # Load the image
    img = cv2.imread(map_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    #refine the image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # #apply mask
    # mask = np.zeros_like(gray)
    # mask[10:-10, 10:-10] = 255

    # Detect corners
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Dilate the corners
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Display the image
    cv2.imshow('Corners', img)
    cv2.waitKey(0)
    if ord('q'):
        cv2.destroyAllWindows()
        sys.exit(0)


    return


if __name__ == "__main__":
    corner_detect('./maps_test/048011.png')
# end main