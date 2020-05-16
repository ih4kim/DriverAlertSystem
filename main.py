import cv2 as cv2
import numpy
import time

if __name__ == "__main__":
    while True:
        image = readFrame()
        eyes_closed = eyeDetection(image)
        timer = 0
        initial_time = time.perf_counter()
        while (eyes_closed and timer < 5.0):
            image = readFrame()
            eyes_closed = eyes_closed(image)
            timer = time.perf_counter() - initial_time
        if (eyes_closed):
            alertSystem()