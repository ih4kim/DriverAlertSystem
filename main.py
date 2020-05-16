import cv2 as cv2
import numpy
import time
import ClosedEyeDetection

def readFrame(video_capture):
    ret, frame = video_capture.read()
    return frame

def alertSystem():
    #Arduino stuff

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)
    while True:
        image = readFrame(video_capture)
        eyes_closed = CloseEyeDetection.eyeDetection(image)
        timer = 0
        initial_time = time.perf_counter()
        while (eyes_closed and timer < 5.0):
            image = readFrame()
            eyes_closed = eyes_closed(image)
            timer = time.perf_counter() - initial_time
        if (eyes_closed):
            alertSystem()