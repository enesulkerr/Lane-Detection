import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import math
import cv2

def gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
def gauss(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)
def canny(frame):
    edges = cv2.Canny(frame,50,150)
    return edges

def region_of_interest(frame):
    height = frame.shape[0]
    polygons = np.array([
    [(200, height), (550,250), (1100, height)]
                        ])
    mask = np.zeros_like(frame)
    mask = cv2.fillPoly(mask, polygons, 255)
    mask = cv2.bitwise_and(frame, mask)
    return mask

def average_slope_intercept(frame,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left_fit.append((slope,y_int))
        else:
            right_fit.append((slope,y_int))

    left_fit_average = np.average(left_fit,axis = 0)
    right_fit_average = np.average(right_fit,axis = 0)
    left_line = make_points(frame,left_fit_average)
    right_line = make_points(frame,right_fit_average)
    return np.array([left_line,right_line])

def make_points(frame,line_parameteres):
    slope,y_int = line_parameteres
    y1 = frame.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - y_int)/slope)
    x2 = int((y2 - y_int)/slope)
    return np.array([x1,y1,x2,y2])

def display_lines(frame,lines):
    line_frame =  np.zeros_like(frame)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_frame,(x1,y1),(x2,y2),(255,0,0),10)
    return line_frame

cap = cv2.VideoCapture('test_video.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    gray_frame  =  gray(frame)
    gauss_frame =  gauss(gray_frame)
    canny_frame =  canny(gauss_frame)
    masked_frame  =  region_of_interest(canny_frame)
    lines = cv2.HoughLinesP(masked_frame,2, np.pi/180,100, np.array([]),minLineLength=40,maxLineGap = 5)
    averaged_frame = average_slope_intercept(frame,lines)
    line_frame = display_lines(frame,averaged_frame)
    strip_frame = cv2.addWeighted(frame,0.8,line_frame,1,1)
    cv2.imshow("result",strip_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
