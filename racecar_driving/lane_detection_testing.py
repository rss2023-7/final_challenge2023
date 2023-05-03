#!/usr/bin/env python

import cv2
import numpy as np
import math


def color_segmentation(img):
    bounding_box = ((0,0),(0,0))

    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 30, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Erode the mask to remove noise
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=2)

    # Dilate the mask to improve the shape
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=2)



    opencv_major_version = int(cv2.__version__.split('.')[0])

    # Call cv2.findContours with the appropriate output format based on the major version
    if opencv_major_version >= 4:
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour
    max_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            bounding_box = ((x, y), (x + w, y + h))

    return bounding_box

def preprocess_image(img):
    # Resize the image if necessary
    img = cv2.resize(img, (640, 480))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges in the image
    edges = cv2.Canny(blur, 50, 150)

    return edges

def detect_lane_lines(edges):
    # Apply HoughLinesP to detect lines in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)

    return lines


image = cv2.imread("../media/test_images/IMG_8485.png")
#display the image
cv2.imshow('original', image)

# # apply and show color segmentation
# bounding_box = color_segmentation(image)
# cv2.rectangle(image, bounding_box[0], bounding_box[1], (0, 255, 0), 5)
# cv2.imshow('color segmentation', image)


hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the color white
# Hue can be any value, so we set a wide range (0-179)
# Low saturation (0-30) and high value (200-255) for white
lower_white = np.array([0, 0, 200])
upper_white = np.array([179, 30, 255])

# Threshold the HSV image to get only white colors
mask = cv2.inRange(hsv_image, lower_white, upper_white)

# Bitwise-AND the mask and the original image
result = cv2.bitwise_and(image, image, mask=mask)


# crop the resulting image to only focus on the right half of the image and the bottom 50%
height, width, _ = result.shape
result_cropped = result[int(height/2):height, int(width/2):width]


hough_image = image.copy()
#crop hough image to only focus on the right half of the image and the bottom 50%
height, width, _ = hough_image.shape
hough_image = hough_image[int(height/2):height, int(width/2):width]

# Canny on original image
canny_original = cv2.Canny(hough_image, 50, 200, None, 3)


# Canny on masked image
canny_masked = cv2.Canny(result, 50, 200, None, 3)

cdst_original = cv2.cvtColor(canny_original, cv2.COLOR_GRAY2BGR)
cdst_masked = cv2.cvtColor(canny_masked, cv2.COLOR_GRAY2BGR)
cdstP_original = np.copy(cdst_original)
cdstP_masked = np.copy(cdst_masked)

#display cdst
cv2.imshow('cdst original', cdst_original)
cv2.imshow('cdst masked', cdst_masked)





def perform_hough_transform(src, dst, cdst, cdstP):

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            #printo rho and theta
            print('here')
            print("rho: ", rho, "theta: ", theta)

            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    
    
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            print(l)
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    
    cv2.imshow("Source", src)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)


#perform hough transform on original image
perform_hough_transform(hough_image, canny_original, cdst_original, cdstP_original)
#perform hough transform on masked image
# perform_hough_transform(hough_image, canny_masked, cdst_masked, cdstP_masked)



# preprocess_image = preprocess_image(hough_image)
# lines = detect_lane_lines(preprocess_image)
# if lines is not None:
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             hough_image = cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 5)

#             #show the image with the lines drawn
#             cv2.imshow("lane lines", hough_image)



# # Display the images
# cv2.imshow('Original Image', image)
# cv2.imshow('HSV Image', hsv_image)
# cv2.imshow('Mask', mask)
# cv2.imshow('Result', result)
# cv2.imshow('Result Cropped', result_cropped)
# #save result cropped
# cv2.imwrite('../media/test_images/IMG_8485_result_cropped.png', result_cropped)



cv2.waitKey(0)



def image_callback(image_message):
    try:
        img = bridge.imgmsg_to_cv2(image_message, "bgr8")
    except CvBridgeError as e:
        print(e)
        return

    # Image preprocessing
    edges = preprocess_image(img)

    # Lane detection
    lines = detect_lane_lines(edges)

    # Visualize the detected lines
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

                #show the image with the lines drawn
                cv2.imshow("lane lines", img)

   








