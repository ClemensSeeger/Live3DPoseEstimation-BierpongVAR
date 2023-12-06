import cv2 as cv
import numpy as np
import time
from movenet.models.model_factory import load_model
from libraries.lib import Image_Stream,Image_Processer,KeypointEstimator, draw_keypoints,calculate_3d_points,draw_3d_points
import h5py
import utils as ut
import matplotlib.pyplot as plt

#Load the File containing the camera-parameters
datafile = h5py.File('camera_calibrationfile_final3.h5',
                             'r')
cmtx0 = datafile['mtx1'][:]
cmtx1 = datafile['mtx2'][:]
dist0 = datafile['dist1'][:]
dist1 = datafile['dist2'][:]
R0 = datafile['R0'][:]
R1 = datafile['R1'][:]
T0 = datafile['T0'][:]
T1 = datafile['T1'][:]
datafile.close()


P0 = cmtx0 @ ut.make_homogeneous_rep_matrix(R0,T0)[:3, :]
P1 = cmtx1 @ ut.make_homogeneous_rep_matrix(R1,T1)[:3, :]



def main():
    filter_length = 5
   # filter = [0.03,-0.07,0.1,-0.136,0.16,0.822,0.16,-0.136,0.1,-0.07,0.03]
    keypoints_1 = np.zeros(shape=(filter_length,17,3))
    keypoints_2 = np.zeros(shape=(filter_length,17,3))
    elbow_3d = np.zeros(shape=(filter_length,6,3))
    filtered_elbow_3d = np.zeros(shape=(6,3))
    elbow = []
    elbow_filtered = []
    fps_list = []
    elbow_z = []
    elbow_z_filtered = []
    counter = 0

    filtered_kp1 = np.zeros(shape=(17,3))
    filtered_kp2 = np.zeros(shape=(17, 3))
    image_streamer = Image_Stream()

    image_processer = Image_Processer(image_streamer,256)

    model = load_model("movenet_thunder", ft_size=64)
    model = model.cuda()

    kp_estimator = KeypointEstimator(model,image_processer)


    while True:
        try:
            start_time = time.time()
            d_view = np.zeros(shape=(480, 480, 3), dtype=np.uint8)


            image1,image2 = image_processer.get_DisplayImages()
            kp1,kp2 = kp_estimator.getKeypoints()

            print('Model FPS: ', kp_estimator.getFPS())
            fps_list.append(kp_estimator.getFPS())
         #   print(kp1.shape, keypoints_1.shape)
            if counter==filter_length:
                counter=0
            if kp1 is not None:
                keypoints_1[counter] = kp1
                elbow.append(kp1[8])
            if kp2 is not None:
                keypoints_2[counter] = kp2


           # keypoints_1.reshape(())
            for i in range(17):
                filtered_kp1[i,0] = np.mean(keypoints_1[:,i,0])
                filtered_kp1[i, 1] = np.mean(keypoints_1[:,i,1])
                filtered_kp1[i, 2] = np.mean(keypoints_1[:,i,2])

                filtered_kp2[i, 0] = np.mean(keypoints_2[:,i,0])
                filtered_kp2[i, 1] = np.mean(keypoints_2[:,i,1])
                filtered_kp2[i, 2] = np.mean(keypoints_2[:,i,2])

            elbow_filtered.append(filtered_kp1[8,1])
            #print(filtered_kp1[8,1])
            object_coord = calculate_3d_points(P0,P1,filtered_kp1,filtered_kp2)

            elbow_3d[counter] = object_coord
            counter = counter + 1
            elbow_z.append(object_coord[2,2])
           # print(object_coord.shape)

            for i in range(6):
                filtered_elbow_3d[i,0] = np.mean(elbow_3d[:,i,0])
                filtered_elbow_3d[i, 1] = np.mean(elbow_3d[:,i,1])
                filtered_elbow_3d[i, 2] = np.mean(elbow_3d[:,i,2])

            elbow_z_filtered.append(filtered_elbow_3d[2,2])
            image1 = draw_keypoints(filtered_kp1,image1,5,11)
            image2 = draw_keypoints(kp2, image2, 5, 11)
            image3 = draw_3d_points(object_coord,d_view)

            together = np.concatenate((image1,image3),axis=1)
            cv.imshow('frame1',together)
            #cv.imshow('frame2', image2)
            #cv.imshow('oben', d_view)
            if cv.waitKey(1) == ord('q'):
                break

            fps = 1 / (time.time() - start_time)

            print(fps)
        except Exception as e:
            print(e)

    elbow = np.asarray(elbow)
    timet = np.arange(0,len(elbow))
   # print(elbow)
    elbow_filtered = np.asarray(elbow_filtered)
    print(elbow.shape)
    print(elbow_filtered.shape)
    print(elbow_filtered)
    elbow_z = np.asarray(elbow_z)
    elbow_z_filtered = np.asarray(elbow_z_filtered)

    print(np.mean(fps_list))
    fig,ax = plt.subplots(2,2)

    ax[0,0].plot(timet,elbow[:,1])

    ax[0,1].plot(timet, elbow_filtered)
    ax[1, 0].plot(timet, elbow_z)
    ax[1, 1].plot(timet, elbow_z_filtered)
    plt.show()

if __name__ == "__main__":
    main()