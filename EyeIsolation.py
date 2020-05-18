import cv2 as cv2
import numpy as np
import pdb


def isolateEye(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    roi_eyes = []
    for (x,y,w,h) in faces:
        roi_gray = image[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        dsize = (80,80)
        for (ex, ey, ew, eh) in eyes:
            resolutionImage = cv2.resize(roi_gray[ey:ey+eh, ex:ex+ew], dsize, interpolation = cv2.INTER_AREA)
            roi_eyes.append(resolutionImage)   
    return roi_eyes
    



