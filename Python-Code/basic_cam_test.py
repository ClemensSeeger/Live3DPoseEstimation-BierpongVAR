import numpy as np
import cv2 as cv
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import threading
import time

#import mediapipe as mp

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

model = hub.load("Model")
model_net = model.signatures['serving_default']

def movenet(input_image):
     # SavedModel format expects tensor type of int32.
  #  input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model_net(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0']
    return keypoints_with_scores


cap1 = cv.VideoCapture(0,cv.CAP_DSHOW)
if not cap1.isOpened():
    print("Cannot open camera")
    exit()

fourcc = cv.VideoWriter_fourcc(*'MJPG')
print(fourcc)


cap1.set(cv.CAP_PROP_FPS,60)
cap1.set(cv.CAP_PROP_FORMAT,-1)
cap1.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap1.set(cv.CAP_PROP_FOURCC,fourcc)


cap2 = cv.VideoCapture(2,cv.CAP_DSHOW)
if not cap2.isOpened():
    print("Cannot open camera")
    exit()

fourcc = cv.VideoWriter_fourcc(*'MJPG')
print(fourcc)


cap2.set(cv.CAP_PROP_FPS,60)
cap2.set(cv.CAP_PROP_FORMAT,-1)
cap2.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv.CAP_PROP_FOURCC,fourcc)

class CamReader:

    def __init__(self,cap):
        self.cap = cap
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):

        while True:
            try:
                start_time = time.time()
                self.ret, self.image = self.cap.read()


                self.frame_cut= self.image[:,80:560]
                self.frame_rgb = cv.cvtColor(self.frame_cut, cv.COLOR_BGR2RGB)
                self.img_format = np.array(Image.fromarray(self.frame_rgb).resize((256,256), 1))
                self.img = np.expand_dims(self.img_format, axis=0)
                self.img_done = tf.cast(self.img, dtype=tf.int32)
                self.fps = 1/(time.time() - start_time)
            except Exception as e:
                print(e)

    def getunprocessed(self):
        return self.frame_cut

    def getImage(self):
        return self.img_done

    def getFPS(self):
        return self.fps

class PoseEstimation:

    def __init__(self,network,cam,cam2):
        self.model = network
        self.cam = cam
        self.cam2 = cam2
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()


    def update(self):

        while True:
            try:
                self.start_time = time.time()
                self.image = self.cam.getImage()
                self.image2 = self.cam2.getImage()
                self.output1 = self.model(self.image)
                self.output2 = self.model(self.image2)
                self.fps = 1/(time.time() - self.start_time)
            except Exception as e:
                print(e)

    def getKeypoints(self):
        return self.output1,self.output2

    def getImage(self):
        return self.image

    def getFPS(self):
        return self.fps



#Start Pose-Estimator Thread



cam1 = CamReader(cap1)
cam2 = CamReader(cap2)

Estimator1 = PoseEstimation(movenet,cam1,cam2)
#Estimator2 = PoseEstimation(movenet,cam2)

while True:
    try:
        start_time = time.time()
        frame1=cam1.getImage()
        frame2 = cam2.getImage()

       # ret,frame1 = cap1.read()
       # ret, frame2 = cap2.read()
       # frame_rgb = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
       # frame_rgb2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
            #frame_rgb = frame
        fps_est1 = Estimator1.getFPS()
        keypoints,keypoints2 = Estimator1.getKeypoints()
        #keypoints = Estimator1.getKeypoints()

       # fps_est2 = Estimator2.getFPS()
        print(fps_est1)
      #  img = np.array(Image.fromarray(frame_rgb).resize((256, 256), 1))
       # img = np.expand_dims(img, axis=0)
        #casted = tf.cast(img, dtype=tf.float32)
       # output = movenet(frame1)
        #output2 = movenet(frame2)

        #output = movenet(img)
        #output2 = movenet(img)



        frame1 = cam1.getunprocessed()
        frame2 = cam2.getunprocessed()
        together=np.concatenate((frame1,frame2),axis=1)

        #keypoints = output['output_0'].numpy()
        # print(keypoints)
        # print(keypoints.shape)
        keypoints = np.reshape(keypoints[:, :, :51], newshape=(1, 6, 17, 3))
        keypoints2 = np.reshape(keypoints2[:, :, :51], newshape=(1, 6, 17, 3))
        #  print(keypoints.shape)
        for point in keypoints[0, 0, 5:11]:
            #    # print(point)
            # print(point.shape)
            x = int(point[0] * 480)
            y = int(point[1] * 480)
            cv.circle(together, (y, x), 5, (255, 255, 255), 2)

        for point in keypoints2[0, 0, 5:11]:
            #    # print(point)
            # print(point.shape)
            x = int(point[0] * 480 )
            y = int(point[1] * 480 +480)
            cv.circle(together, (y, x), 5, (255, 125, 255), 2)

        #print(output)
        #print(casted)
        cv.imshow('frame',together)
        if cv.waitKey(1) == ord('q'):
            break

        fps = 1 / (time.time() - start_time)
        print('Calculated fps: ', fps)
        #print('cam1: ',cam1.getFPS())
        #print('cam2: ',cam2.getFPS())
    except Exception as e:
        print(e)


# When everything done, release the capture
cap1.release()
cap2.release()
cv.destroyAllWindows()