#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge, CvBridgeError

# Initialize the node and create a publisher for processed images
rospy.init_node('lane_detection_node')
processed_image_pub = rospy.Publisher('/processed_image', Image, queue_size=1)
bridge = CvBridge()

# Global variable to store the current lane assignment
current_lane_assignment = 0

# Callback function to update the lane assignment
def lane_assignment_callback(msg):
    global current_lane_assignment
    current_lane_assignment = msg.data

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

def segment_lane_lines(lines, lane_assignment):
    # Filter the lines based on slope, length, and position
    # Customize the filtering criteria based on your lane assignment
    segmented_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 < lane_assignment * 100 and x2 < (lane_assignment + 1) * 100:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5:
                    segmented_lines.append(line)

    return segmented_lines

def image_callback(msg):
    global current_lane_assignment
    try:
        img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
        return

    # Image preprocessing
    edges = preprocess_image(img)

    # Lane detection
    lines = detect_lane_lines(edges)

    if lines is not None:
        # Lane segmentation
        segmented_lines = segment_lane_lines(lines, current_lane_assignment)

        # Draw segmented lines on the original image
        for line in segmented_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Publish the processed image
    try:
        processed_image_pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))
    except CvBridgeError as e:
        print(e)

def main():
    # Subscribe to the raw image and lane assignment topics
    rospy.Subscriber('/camera/image_raw', Image, image_callback)
    rospy.Subscriber('/lane_assignment', Int32, lane_assignment_callback)

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    main()
