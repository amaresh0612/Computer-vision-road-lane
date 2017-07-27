# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:37:27 2017

@author: AMARESH
"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys

import cv2
image = cv2.imread("clouds.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Over the Clouds", image)
cv2.imshow("Over the Clouds - gray", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

image = mpimg.imread('input.jpeg')
grayscaled = grayscale(image)
plt.imshow(grayscaled, cmap = 'gray')

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
  
# apply gaussian blur
kernelSize = 5
gaussianBlur = gaussian_blur(grayscaled, kernelSize)

plt.imshow(gaussianBlur, cmap = 'gray')

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

# canny
minThreshold = 100
maxThreshold = 200
edgeDetectedImage = canny(gaussianBlur, minThreshold, maxThreshold)
plt.imshow(edgeDetectedImage, cmap = 'gray')

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with 
    #depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
  
#apply mask
lowerLeftPoint = [130, 540]
upperLeftPoint = [410, 350]
upperRightPoint = [570, 350]
lowerRightPoint = [915, 540]

pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, 
lowerRightPoint]], dtype=np.int32)
masked_image = region_of_interest(edgeDetectedImage, pts)
plt.imshow(masked_image, cmap = 'gray')



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
              minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    
    draw_lines(line_img, lines)
    return line_img

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.    
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    
#hough lines
rho = 1
theta = np.pi/180
threshold = 30
min_line_len = 20 
max_line_gap = 20

houged = hough_lines(masked_image, rho, theta, 
                  threshold, min_line_len, max_line_gap)
plt.imshow(houged, cmap = 'gray')


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.    
    """
    imshape = img.shape
    
    # these variables represent the y-axis coordinates to which 
    # the line will be extrapolated to
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]
    
    # left lane line variables
    all_left_grad = []
    all_left_y = []
    all_left_x = []
    
    # right lane line variables
    all_right_grad = []
    all_right_y = []
    all_right_x = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            gradient, intercept = np.polyfit((x1,x2), (y1,y2), 1)
            ymin_global = min(min(y1, y2), ymin_global)
            
            if (gradient > 0):
                all_left_grad += [gradient]
                all_left_y += [y1, y2]
                all_left_x += [x1, x2]
            else:
                all_right_grad += [gradient]
                all_right_y += [y1, y2]
                all_right_x += [x1, x2]
    
    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)
    
    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)
    
    # Make sure we have some points in each lane line category
    if ((len(all_left_grad) > 0) and (len(all_right_grad) > 0)):
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

        cv2.line(img, (upper_left_x, ymin_global), 
                      (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global), 
                      (lower_right_x, ymax_global), color, thickness)

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
  
# outline the input image
colored_image = weighted_img(houged, image)   

plt.imshow(colored_image) 


from moviepy.editor import VideoFileClip
from IPython.display import HTML


import imageio
imageio.plugins.ffmpeg.download()



def process_image(image):
    # grayscale the image
    grayscaled = grayscale(image)

    # apply gaussian blur
    kernelSize = 5
    gaussianBlur = gaussian_blur(grayscaled, kernelSize)

    # canny
    minThreshold = 100
    maxThreshold = 200
    edgeDetectedImage = canny(gaussianBlur, minThreshold, maxThreshold)

    # apply mask
    lowerLeftPoint = [130, 540]
    upperLeftPoint = [410, 350]
    upperRightPoint = [570, 350]
    lowerRightPoint = [915, 540]

    pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, 
                    lowerRightPoint]], dtype=np.int32)
    masked_image = region_of_interest(edgeDetectedImage, pts)

    # hough lines
    rho = 1
    theta = np.pi/180
    threshold = 30
    min_line_len = 20 
    max_line_gap = 20

    houged = hough_lines(masked_image, rho, theta, threshold, min_line_len, 
                         max_line_gap)

    # outline the input image
    colored_image = weighted_img(houged, image)
    return colored_image

output = 'car_lane_detection.mp4'
clip1 = VideoFileClip("challenge.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(output, audio=False)