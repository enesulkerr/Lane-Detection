import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import math
import cv2

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
canny = cv2.Canny(blur,50,150)


plt.imshow(canny)
plt.show()
