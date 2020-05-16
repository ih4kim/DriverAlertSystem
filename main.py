import cv2 as cv2
import numpy
import time
#import ClosedEyeDetection

def readFrame(video_capture):
    ret, frame = video_capture.read()
    return frame

#def alertSystem():
    #Arduino stuff

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)
    start = True
    while (start):
        image = readFrame(video_capture)
        cv2.imshow("Video", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            start = False
            break
        #eyes_closed = CloseEyeDetection.eyeDetection(image)
        timer = 0
        #initial_time = time.perf_counter()
        eyes_closed = True
        while (eyes_closed and timer < 5.0):
            image = readFrame(video_capture)
            cv2.imshow("Video", image)
            #eyes_closed = CloseEyeDetection.eyeDetection(image)
            #timer = time.perf_counter() - initial_time
            if cv2.waitKey(1) & 0xFF == ord('q'):
                start = False
                break
        if (eyes_closed):
            pass #add arduino code
