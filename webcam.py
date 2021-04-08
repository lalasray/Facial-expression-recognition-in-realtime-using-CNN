# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:41:43 2018

@author: Madhusudhan
"""

import numpy as np
import cv2
import argparse
import os, sys
from keras.preprocessing import image

#sys.path.append("../")

import face_detection as fd

import cnn1 as cnn

windowName = 'Preview Screen'
face_shape = (128, 128)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']


parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('-testImage', help=('Given the path of testing image, the program will predict the result of the image.'
"This function is used to test if the model works well."))

args = parser.parse_args()

model = cnn.load_model("cnn_128x128_2convmax_2fc_64f_32bs_100e.h5")
#cnn6class_64x64_2convmax_2fc_64f_64bs_60e_wdropoutp5
#cnn_64x64_2convmax_2fc512neu_64f_64bs_60ge_wdropout
def refreshFrame(frame, startX, startY, endX, endY, index):
    
    if startX is not None:
        fd.draw_rect(frame, startX, startY, endX, endY, index)
    cv2.imshow(windowName, frame)


def display_and_classify(capture):
    i=1
    while (True):
        flag, frame = capture.read()
        """
        cv2.imshow(windowName+"s", frame)
        if cv2.waitKey(1) == 27:
                break
        """
        #faceCoordinates = None
        if i==1:
            index = 4
            i -= 1
            faceCoordinates=None
        
        try:
            faceCoordinates = fd.face_detect(frame)
            startX = faceCoordinates[0]
            startY = faceCoordinates[1]
            endX = faceCoordinates[2]
            endY = faceCoordinates[3]
            #faceCoordinates = dict['faceCoordinates']
            #text = dict['text']
            refreshFrame(frame, startX, startY, endX, endY, index)
        except:
            refreshFrame(frame, None, None, None, None, 7)
        
        if faceCoordinates is not None:
            face_img = fd.face_crop(frame, faceCoordinates, face_shape=face_shape)
            #cv2.imshow(windowsName, face_img)
            cv2.imwrite('testing.png', face_img)
            im = image.load_img('testing.png', target_size = (128, 128))
            im = image.img_to_array(im)
            im = np.expand_dims(im, axis = 0)            

            result = model.predict(im)        #[0]
            index = np.argmax(result)
            print(emotions[index])       #, 'prob:', max(result))
            print("")
            #text = "{:.2f}%".format(emotions[index] * 100)
            
            # print(face_img.shape)
            # emotion = class_label[result_index]
            # print(emotion)
            
            if cv2.waitKey(1) == 27:
                break  # esc to quit
                break
            
def getCameraStream():
    capture = cv2.VideoCapture(0)
    if not capture:
        print("Failed to capture video streaming ")
        sys.exit(1)
    else:
        print("Successfully captured video stream")
        
    return capture

def main():
    '''
    Arguments to be set:
        showCam : determine if show the camera preview screen.
    '''
    print("Enter main() function")
    
    if args.testImage is not None:
        img = cv2.imread(args.testImage)
        faceCoordinates = fd.face_detect(img)
        face_img = fd.face_crop(img, faceCoordinates, face_shape=face_shape)
        #cv2.imshow(windowsName, face_img)
        cv2.imwrite('testing.png', face_img)
        im = image.load_img('testing.png', target_size = (128, 128))
        im = image.img_to_array(im)
        im = np.expand_dims(im, axis = 0)

        result = model.predict(im)        #[0]
        index = np.argmax(result)
        print(emotions[index])            #, 'prob:', max(result))
        sys.exit(0)

    showCam = 1

    capture = getCameraStream()

    if showCam:
        cv2.startWindowThread()
        cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
    
    display_and_classify(capture)

if __name__ == '__main__':
    main()
