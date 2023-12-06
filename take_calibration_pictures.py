from libraries.lib import Image_Stream
import cv2 as cv


image_streamer = Image_Stream()

# Image left
image_path_l = '.\Images\Left\calibration_l_'
# Image right
image_path_r = '.\Images\Right\calibration_r_'
# Image together
image_path_t = '.\Images\Together\calibration_'

counter=0
while True:
    try:


        image1,image2 = image_streamer.get_Images()

        cv.imshow('frame1', image1)
        cv.imshow('frame2', image2)
        if cv.waitKey(10) == ord('q'):
            break

        if cv.waitKey(1) == ord('s'):
            counter = counter + 1
            filename_l = image_path_l  + str(counter) + '.jpg'
            filename_r = image_path_r + str(counter) + '.jpg'

            filename_t_l = image_path_t + 'l_' + str(counter) + '.jpg'
            filename_t_r = image_path_t + 'r_' + str(counter) + '.jpg'


            cv.imwrite(filename_l, image1)
            cv.imwrite(filename_r, image2)

            cv.imwrite(filename_t_l, image1)
            cv.imwrite(filename_t_r, image2)
            print('images saved')

    except Exception as e:
        print(e)