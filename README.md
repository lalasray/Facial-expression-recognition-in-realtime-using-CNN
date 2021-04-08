# Facial-expression-recognition-in-realtime-using-CNN
Real Time Facial Expression Recognition from Webcam using Convolutional Neural Networks and OpenCV

This projects uses Keras with Tensorflow Backend and OpenCV to develop a Convolutional Neural Network which can classify, from an image or from a real time video feed (webcam), the emotion/expression that a person is showing. The CNN was trained on 90% of a combination of the CK+, JAFFE and the KDEF datasets for 40 epochs. The remaining 10% was used as the test set to check the validation accuracy of the model (70%).

Capture of the live video feed from the webcam and processing that video feed was done using OpenCV. Face detection was implemented using the dnn module of OpenCV.

#Run webcam.py from the command prompt to get FER in real time from webcam.
