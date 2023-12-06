import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py

def calibrate_camera(images_folder):
    images_names = sorted(glob.glob(images_folder))
    images = []
    print(images_names)
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 6 # number of checkerboard rows.
    columns = 9  # number of checkerboard columns.
    world_scaling = 4.7 # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    print(width,height)
    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #plt.imshow(gray)
        #plt.show()
        # find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        print(ret,corners)
        if ret == True:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (3, 3)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.imshow('img', frame)
            k = cv.waitKey(200)

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)

    return ret,mtx, dist


ret1, mtx1, dist1 = calibrate_camera(images_folder='Images/Left/*')
ret2, mtx2, dist2 = calibrate_camera(images_folder='Images/Right/*')

print(ret1,ret2)
print(mtx1,mtx2)


def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    # read the synched frames
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names) // 2]
    c2_images_names = images_names[len(images_names) // 2:]
    print(c1_images_names)
    print(c2_images_names)
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)

    # change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = 6  # number of checkerboard rows.
    columns = 9# number of checkerboard columns.
    world_scaling = 4.7  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (3, 3), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (3, 3), (-1, -1), criteria)

            cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
            cv.imshow('img', frame1)

            cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
            cv.imshow('img2', frame2)
            k = cv.waitKey(20)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1,
                                                                 dist1,
                                                                 mtx2, dist2, (width, height), criteria=criteria,
                                                                 flags=stereocalibration_flags)

    print(ret)
    return R, T


R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'Images/Together/*')


print(R,T)


def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1

    return P


# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]
    return P


R1 = R; T1 = T
R0 = np.eye(3, dtype=np.float32)
T0 = np.array([0., 0., 0.]).reshape((3, 1))

datafile = h5py.File('camera_calibrationfile_final3.h5',
                             'w')
datafile.create_dataset('mtx1', data=mtx1)
datafile.create_dataset('mtx2', data=mtx2)
datafile.create_dataset('dist1', data=dist1)
datafile.create_dataset('dist2', data=dist2)
datafile.create_dataset('R0', data=R0)
datafile.create_dataset('R1', data=R1)
datafile.create_dataset('T0', data=T0)
datafile.create_dataset('T1', data=T1)
