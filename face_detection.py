# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:19:42 2018

@author: Madhusudhan
"""

# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
#import argparse
import cv2

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
#index = 4


"""
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
"""


def face_detect(im):
    # load our serialized model from disk
    #print("[INFO] loading face detection model...")
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    #image = cv2.imread('testt3.png')
    
    image = im
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
    	(300, 300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the detections and
    # predictions
    #print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with the
    	# prediction
    	confidence = detections[0, 0, i, 2]
    
    	# filter out weak detections by ensuring the `confidence` is
    	# greater than the minimum confidence
    	if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")  
            faceCoordinates = box.astype("int")
            
            #text = "{:.2f}%".format(confidence * 100)
            
    return faceCoordinates
    

    # show the output image
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)
    
def draw_rect(image, startX, startY, endX, endY, index):
    # draw the bounding box of the face along with the associated probability
    #text = "{:.2f}%".format(conf * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(image, emotions[index], (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)    
    
def face_crop(image, faceCoordinates, face_shape):
    face = crop_face(image, faceCoordinates)            
    face_scaled = cv2.resize(face, face_shape)
    face_gray = cv2.cvtColor(face_scaled, cv2.COLOR_BGR2GRAY)
    return face_gray

def crop_face(img, faceCoordinates):
    return img[faceCoordinates[1]:faceCoordinates[3], faceCoordinates[0]:faceCoordinates[2]]