import numpy as np
import cv2 as cv
import pygame
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import threading
import time
from tensorflow.python.framework.ops import disable_eager_execution




print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
print('hello world')
gpus = tf.config.list_physical_devices('GPU')
#tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

model = hub.load("Model")
#module = hub.Module('Model2/')
#model_pb = tf.saved_model.load('Model2/movenet_multipose_lightning_1.tar.gz')
#print(model_pb)
#model = tf.keras.models.load_model("https://tfhub.dev/google/movenet/multipose/lightning/1")
print(model)

print(model.signatures)
movenet = model.signatures['serving_default']

print(movenet)
#model2 = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
#movenet2 = model2.signatures['serving_default']

cap = cv.VideoCapture(0,cv.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

fourcc = cv.VideoWriter_fourcc(*'MJPG')
print(fourcc)


cap.set(cv.CAP_PROP_FPS,60)
cap.set(cv.CAP_PROP_FORMAT,-1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv.CAP_PROP_FOURCC,fourcc)

#cap.set(cv.CAP_DSHOW, 1)



for i in range(30):
    print(cap.get(i))



pygame.init()
screen_w = 1080
screen_h = 720
try:
    screen = pygame.display.set_mode((screen_w, screen_h), flags=pygame.RESIZABLE)
except TypeError:
    screen = pygame.display.set_mode((screen_w, screen_h))
pygame.display.set_caption("Frame Buffer")
clock = pygame.time.Clock()

startcheck=False


class PoseEstimation:

    def __init__(self,network,cap):
        self.model = network
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()


    def update(self):

        while True:
            try:
                start_time = time.time()
                self.ret, self.image = cap.read()

                self.frame_rgb = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
                self.img = np.array(Image.fromarray(self.frame_rgb).resize((256, 256), 1))
                self.img = np.expand_dims(self.img, axis=0)
                self.output1 = self.model(tf.cast(self.img, dtype=tf.int32))['output_0']
                #self.output2 = self.model(tf.cast(self.img, dtype=tf.int32))
                self.fps = 1/(time.time() - start_time)
            except Exception as e:
                print(e)

    def getKeypoints(self):
        return self.output1.numpy()

    def getImage(self):
        return self.frame_rgb

    def getFPS(self):
        return self.fps


i=0
table_left = [0,0,0]
table_right = [0,0,0]

@tf.function
def estimate_pose(image):
    output = movenet(image)
    keypoints = output['output_0']

    return keypoints

#Start Pose-Estimator Thread
Estimator1 = PoseEstimation(movenet,cap)
#Estimator2 = PoseEstimation(movenet,cap)

while True:
    try:
        start_time = time.time()
        ret, frame = cap.read()
        #Limit to max. 60FPS (outdated)
        clock.tick()

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb = frame_rgb[:, 140:500]
       # frame_rgb = np.concatenate((frame_rgb,frame_rgb),axis=1)
        print(frame_rgb.shape)
        #output = Estimator1.getKeypoints()
        #frame_rgb = Estimator1.getImage()

        #output2 = Estimator2.getKeypoints()
        img_out = frame_rgb.copy()
        if startcheck == True:
            #output1,output2 = Estimator.getKeypoints()


            img = np.array(Image.fromarray(frame_rgb).resize((256,256),1))

            img = np.expand_dims(img,axis=0)
            #img = np.expand_dims(img, axis=0)

           # images = tf.convert_to_tensor(np.concatenate((img,img),axis=0),dtype=tf.int32)
           # print(images)

            #outputs=tf.vectorized_map(estimate_pose,elems=images,fallback_to_while_loop=False,warn=False)
            #print(outputs)
           # output = estimate_pose(images)
           # output2 = estimate_pose(tf.cast(img, dtype=tf.int32))
           # output = movenet(tf.cast(img, dtype=tf.int32))
           # output2 = movenet(tf.cast(img, dtype=tf.int32))
            output=Estimator1.getKeypoints()
            #print('Thread FPS: ',Estimator1.getFPS())
            #print(output)
           # print('Thread FPS: ', Estimator2.getFPS())
            keypoints = output['output_0'].numpy()
           # print(keypoints)
           # print(keypoints.shape)
            keypoints=np.reshape(keypoints[:,:,:51],newshape=(1,6,17,3))
          #  print(keypoints.shape)
            for point in keypoints[0,0,:11]:
           #    # print(point)
                #print(point.shape)
                x = int(point[0::3] * 360)
                y = int(point[1::3] * 360)
                cv.circle(img_out,(y,x),5,(255,255,255),2)


        frame_rot = np.rot90(img_out)

        frame_final = np.flip(frame_rot, axis=0)

        screen.blit(pygame.surfarray.make_surface(frame_final), (0, 0))
       # fps = 1 / (time.time() - start_time)
       # print('Calculated fps: ', fps)
        print(clock.get_fps())
    except Exception as e:
        print(e)
    # finally update the display
    pygame.display.update()

    FPS = clock.get_fps()

    for event in pygame.event.get():

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                startcheck = False
            if event.key == pygame.K_UP:
                startcheck = True
            if event.key == pygame.K_LEFT:
                pygame.quit()
