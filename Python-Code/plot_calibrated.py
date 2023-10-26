import io
import struct
import socket
import pygame
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import threading
import h5py
from utils import DLT, get_projection_matrix, write_keypoints_to_disk




datafile = h5py.File('camera_calibrationfile_final2.h5',
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


print('R0')
print(R0)
print('R1')
print(R1)

print('T0')
print(T0)
print('T1')
print(T1)
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

print(cmtx0,cmtx1)
print('')
print(dist0,dist1)
print('')
P0 = get_projection_matrix(cmtx0, R0, T0)
P1 = get_projection_matrix(cmtx1, R1, T1)

print('P0')
print(P0)
print('P1')
print(P1)

class Image_Stream:
    def __init__(self):
        self.socket=socket.socket()
        self.socket.bind(('192.168.178.60',8000))
        self.socket.listen()
        self.connection = self.socket.accept()[0].makefile('rb')
        self.thread = threading.Thread(target=self.update,daemon=True)
        self.thread.start()
    def update(self):
        while True:

            self.image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]

            if self.image_len is None:
                break
            self.image1 = self.connection.read(self.image_len)

            self.image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
            if self.image_len is None:
                break

            self.image2 = self.connection.read(self.image_len)

    def get_Images(self):
        return self.image1,self.image2







pygame.init()
screen_w = 640*2
screen_h = 480
try:
    screen = pygame.display.set_mode((screen_w, screen_h), flags=pygame.RESIZABLE)
except TypeError:
    screen = pygame.display.set_mode((screen_w, screen_h))
pygame.display.set_caption("Frame Buffer")
clock = pygame.time.Clock()


image_stream = Image_Stream()

# define coordinate axes in 3D space. These are just the usual coorindate vectors
coordinate_points = np.array([[0., 0., 0.],
                              [1., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 1.]])
z_shift = np.array([0., 0., 230.]).reshape((1, 3))
# increase the size of the coorindate axes and shift in the z direction
draw_axes_points = 20* coordinate_points + z_shift

# project 3D points to each camera view manually. This can also be done using cv.projectPoints()
# Note that this uses homogenous coordinate formulation
pixel_points_camera0 = []
pixel_points_camera1 = []
for _p in draw_axes_points:
    X = np.array([_p[0], _p[1], _p[2], 1.])

    # project to camera0
    uv = P0 @ X
    uv = np.array([uv[0], uv[1]]) / uv[2]
    pixel_points_camera0.append(uv)

    # project to camera1
    uv = P1 @ X
    uv = np.array([uv[0], uv[1]]) / uv[2]
    pixel_points_camera1.append(uv)

# these contain the pixel coorindates in each camera view as: (pxl_x, pxl_y)
pixel_points_camera0 = np.array(pixel_points_camera0)
pixel_points_camera1 = np.array(pixel_points_camera1)

print(pixel_points_camera0,pixel_points_camera1)

# follow RGB colors to indicate XYZ axes respectively
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
# draw projections to camera0

#pixel_points_camera1 = np.array(pixel_points_camera1)
while True:
    try:

        clock.tick(60)

        image1,image2 = image_stream.get_Images()


        image1_arr = np.array(Image.open(io.BytesIO(image2)))
        image2_arr = np.array(Image.open(io.BytesIO(image1)))


        origin0 = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv2.line(image1_arr, origin0, _p, col, 2)

        origin1 = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv2.line(image2_arr, origin1, _p, col, 2)


        image1_arr = np.rot90(image1_arr)
        image2_arr = np.rot90(image2_arr)

        image1_arr = np.flip(image1_arr,axis=0)
        image2_arr = np.flip(image2_arr, axis=0)


        screen.blit(pygame.surfarray.make_surface(image1_arr),(0,0))
        screen.blit(pygame.surfarray.make_surface(image2_arr),(640, 0))
        pygame.display.update()

        print(clock.get_fps())

    except Exception as e:
        print(e)



    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            image_stream.connection.close()
            image_stream.socket.close()
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_LEFT:
                    bodyparts_offset = bodyparts_3d
