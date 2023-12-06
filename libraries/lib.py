import cv2
import threading
import time
import socket
import pickle
import torch
import numpy as np
from utils import DLT

class Image_Stream:
    def __init__(self,ip = "192.168.178.60" ,port = 8000):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ip = ip
        self.port = port
        self.s.bind((self.ip, self.port))
        self.thread = threading.Thread(target=self.update,daemon=True)
        self.thread.start()
    def update(self):
        while True:
            self.start_time = time.time()

            self.x1 = self.s.recvfrom(10000000)
            self.x2 = self.s.recvfrom(10000000)

            self.clientip1 = self.x1[1][0]
            self.data1 = self.x1[0]
            self.data1 = pickle.loads(self.data1)
            self.image1 = cv2.imdecode(self.data1, cv2.IMREAD_COLOR)

            self.clientip2 = self.x2[1][0]
            self.data2 = self.x2[0]
            self.data2 = pickle.loads(self.data2)
            self.image2 = cv2.imdecode(self.data2, cv2.IMREAD_COLOR)

            self.fps = 1 / (time.time() - self.start_time)

    def get_Images(self):
        return self.image1,self.image2

    def getFPS(self):
        return self.fps


class Image_Processer:

    def __init__(self, image_streamer,size):
        self.image_streamer = image_streamer
        self.size = size
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):

        while True:
            try:

                self.image1, self.image2 = self.image_streamer.get_Images()

                self.image1_proc, self.image1_display = preprocess_image(self.image1,self.size)
                self.image2_proc, self.image2_display = preprocess_image(self.image2, self.size)

                self.image1_proc_tensor = torch.Tensor(self.image1_proc).cuda()
                self.image2_proc_tensor = torch.Tensor(self.image2_proc).cuda()


            except Exception as e:
                print(e)


    def get_DisplayImages(self):
        return self.image1_display, self.image2_display

    def get_processedImages(self):
        return self.image1_proc_tensor,self.image2_proc_tensor

    def getFPS(self):
        return self.fps


class KeypointEstimator:

    def __init__(self,model,im_processor):
        self.model = model
        self.image_processor = im_processor
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):

        while True:

            try:
                self.start_time = time.time()

                self.image1 , self.image2 = self.image_processor.get_processedImages()

                with torch.no_grad():
                    self.kpt_with_conf1 = self.model(self.image1)[0, 0, :, :]
                    self.kpt_with_conf2 = self.model(self.image2)[0, 0, :, :]

                    self.kpt_with_conf1_np = self.kpt_with_conf1.cpu().numpy()
                    self.kpt_with_conf2_np = self.kpt_with_conf2.cpu().numpy()


                self.fps = 1 / (time.time() - self.start_time)
            except Exception as e:
                print(e)

    def getKeypoints(self):
        return self.kpt_with_conf1_np,self.kpt_with_conf2_np


    def getFPS(self):
        return self.fps




def preprocess_image(source_img,size):

    processed_img = cv2.resize(source_img,(size,size),interpolation=cv2.INTER_LINEAR)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    processed_img = processed_img.reshape(1,size,size,3)

    return processed_img, source_img


def draw_keypoints(keypoint_list,image,start_ind,end_ind):
    if 0<start_ind<17:
        if end_ind>start_ind and end_ind<17:
            for i in range(start_ind, end_ind):
                x = int(np.round(keypoint_list[i, 0] * image.shape[0]))
                y = int(np.round(keypoint_list[i, 1] * image.shape[1]))
                cv2.circle(image, (y, x), 2, (255, 255, 255), 2)
    else:
        raise ValueError("Error: starting index or ending index too big/small")

    return image

def calculate_3d_points(P0,P1,keypoint_list1,keypoint_list2,start_ind=5,end_ind=11,size_x=480,size_y=480):



    if 0 < start_ind < 17:
        if end_ind > start_ind and end_ind < 17:
            n_points = end_ind - start_ind
            object_coords = np.zeros(shape=(n_points, 3))

            for i in range(n_points):
                x1 = keypoint_list1[ start_ind+i, 0] * size_x
                y1 = keypoint_list1[ start_ind+i, 1] * size_y

                x2 = keypoint_list2[ start_ind + i, 0] * size_x
                y2 = keypoint_list2[ start_ind + i, 1] * size_y


                object_coords[i] = DLT(P0,P1,[y1,x1],[y2,x2])



    else:
        raise ValueError("Error: starting index or ending index too big/small")

    return object_coords

def draw_3d_points(object_coords,image):
    for i in range(len(object_coords)):
        x = int(object_coords[i, 0]+100) *2
        y = int(object_coords[i, 2]-80) * 2
        cv2.circle(image, (y, x), 2, (255, 50, 50), 2)
    return image