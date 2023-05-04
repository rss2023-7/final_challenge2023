#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge, CvBridgeError

class LaneDetector:

    def __init__(self):
        # Initialize the node and create a publisher for processed images
        rospy.init_node('lane_detection_node')
        self.bridge = CvBridge()
        # Subscribe to the raw image
        rospy.Subscriber('/zed/zed_node/rgb/image_rect_color', Image, self.image_callback)
        #Pulish Line Coordinates
        self.line_coordinates_pub = rospy.Publisher('/line_coordinates', Int32, queue_size=1)

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        cv2.imshow('original', image)
        return image


    def perform_hough_transform(self, src, dst, cdst, cdstP):
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        cv2.imshow("Source", src)
        cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
        return linesP

    def preprocess_image(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the color white
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([179, 30, 255])

        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # Bitwise-AND the mask and the original image
        result = cv2.bitwise_and(image, image, mask=mask)
        return result
    def crop_image(self, image):
        height, width, _ = image.shape
        return image[int(height/2):height, int(width/2):width]

    def perform_canny_edge_detection(self, image):

        # Canny on masked image
        canny_masked = cv2.Canny(image, 50, 200, None, 3)

        cdst_masked = cv2.cvtColor(canny_masked, cv2.COLOR_GRAY2BGR)

        cdstP_masked = np.copy(cdst_masked)

        return canny_masked, cdst_masked, cdstP_masked


    def image_callback(self, image_message):
        try:
            image = self.bridge.imgmsg_to_cv2(image_message, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        preprocessed_image = self.preprocess_image(image)
        result_cropped = self.crop_image(preprocessed_image)
        canny_masked, cdst_masked, cdstP_masked = self.perform_canny_edge_detection(result_cropped)

        #publish line coordinates output from perform_hough_transform function to /line_coordinates topic
        self.line_coordinates_pub.publish(self.perform_hough_transform(result_cropped, canny_masked, cdst_masked, cdstP_masked))




   






if __name__ == '__main__':
    try:
        lane_detector = LaneDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

