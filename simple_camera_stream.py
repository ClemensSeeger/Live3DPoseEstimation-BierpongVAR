import cv2 as cv
import numpy as np
import threading
import time
import torch


# Basic Script for Streaming 2 Images from Pi to PC


from libraries.lib import Image_Stream


image_streamer = Image_Stream()

while True:
    try:
        start_time = time.time()

        image1,image2 = image_streamer.get_Images()

        cv.imshow('frame1', image1)
        cv.imshow('frame2', image2)
        if cv.waitKey(1) == ord('q'):
            break

        fps = 1 / (time.time() - start_time)

        print(fps)
    except Exception as e:
        print(e)