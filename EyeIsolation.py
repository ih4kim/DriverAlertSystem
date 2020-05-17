import cv2 as cv2
import numpy as np
import pdb


def isolateEye(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    roi_eyes = []
    for (x,y,w,h) in faces:
        #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = image[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0),2)
            #if (eh >70 & ew > 70):
            roi_eyes.append(roi_gray[ey:ey+eh, ex:ex+ew])
            #cv2.imshow("Eye", roi_eyes[0])
    return roi_eyes
    



