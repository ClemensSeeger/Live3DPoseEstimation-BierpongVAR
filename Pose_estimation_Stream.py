import cv2 as cv
import numpy as np
import time
from movenet.models.model_factory import load_model

from libraries.lib import Image_Stream,Image_Processer,KeypointEstimator, draw_keypoints




def main():
    filter_length = 5
    keypoints_1 = np.zeros(shape=(filter_length,17,3))
    keypoints_2 = np.zeros(shape=(filter_length, 17, 3))

    counter = 0

    image_streamer = Image_Stream()

    image_processer = Image_Processer(image_streamer,256)

    model = load_model("movenet_thunder", ft_size=64)
    model = model.cuda()

    kp_estimator = KeypointEstimator(model,image_processer)


    while True:
        try:
            start_time = time.time()

            image1,image2 = image_processer.get_DisplayImages()

            kp1,kp2 = kp_estimator.getKeypoints()

            print('Model FPS: ', kp_estimator.getFPS())

            if counter==filter_length:
                counter=0

            if kp1 is not None:
                keypoints_1[counter] = kp1

            if kp2 is not None:
                keypoints_2[counter] = kp2
                counter= counter +1




            image1 = draw_keypoints(keypoints_1,image1,5,11)
            image2 = draw_keypoints(keypoints_2, image2, 5, 11)


            cv.imshow('frame1',image1)
            cv.imshow('frame2', image2)
            if cv.waitKey(1) == ord('q'):
                break

            fps = 1 / (time.time() - start_time)

            print(fps)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()