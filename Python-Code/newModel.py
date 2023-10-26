import io
import struct
import socket
import pygame
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import threading
import h5py
from utils import DLT
from playsound import playsound


#Load the File containing the camera-parameters
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

P0 = get_projection_matrix(cmtx0, R0, T0)
P1 = get_projection_matrix(cmtx1, R1, T1)

print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
print('hello world')
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=6144)])

model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']


# Class for the Image Stream, to receive the images on an extra thread
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
            package_size = struct.calcsize('<L')
            package = self.connection.read(package_size)
            self.image_len = struct.unpack('<L', package)[0]
            if self.image_len is None:
                break
            self.image1 = self.connection.read(self.image_len)

            self.image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
            if self.image_len is None:
                break

            self.image2 = self.connection.read(self.image_len)

    def get_Images(self):
        return self.image1,self.image2






#Initialise the Display
pygame.init()
screen_w = 640*2
screen_h = 480*2
try:
    screen = pygame.display.set_mode((screen_w, screen_h), flags=pygame.RESIZABLE)
except TypeError:
    screen = pygame.display.set_mode((screen_w, screen_h))
pygame.display.set_caption("Frame Buffer")
clock = pygame.time.Clock()

startcheck=False

#start the image-stream Thread
image_stream = Image_Stream()

#Define Class for running the estimation on an extra Thread
class PoseEstimation:

    def __init__(self,network):
        self.model = network
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):

        while True:
            try:

                self.image1,self.image2 = image_stream.get_Images()

                self.image1_arr = np.array(Image.open(io.BytesIO(self.image1)).resize((256, 256), 1))
                self.image1_arr = np.expand_dims(self.image1_arr, axis=0)

                self.output1 = self.model(tf.cast(self.image1_arr, dtype=tf.int32))

                self.image2_arr = np.array(Image.open(io.BytesIO(self.image2)).resize((256, 256), 1))
                self.image2_arr = np.expand_dims(self.image2_arr, axis=0)

                self.output2 = self.model(tf.cast(self.image2_arr, dtype=tf.int32))
            except Exception as e:
                print(e)

    def getKeypoints(self):
        return self.output1,self.output2, self.image1,self.image2


i=0
table_left = [0,0,0]
table_right = [0,0,0]

#Start Pose-Estimator Thread
Estimator = PoseEstimation(movenet)

z_list_el = []
z_list_hand = []
m=0
c=0

line_list=[]

#Functions for checking the Game-Rule
def calculate_line_points(point1,point2):

    x_coords, z_coords = zip([point1[0],point1[2]],[point2[0],point2[2]])
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(A, z_coords)[0]
    print("Line Solution is z = {m}x + {c}".format(m=m, c=c))
    return m,c

def plot_table_line(c,m,table_l,table_r):
    x_len = range(abs(table_r[0]-table_l[0]))
    x_list=[]
    z_list = []
    for i in x_len:
        x = table_l[0]+i
        z = int(m * x + c-100)
        x_list.append(x)
        z_list.append(z)

    return [z_list,x_list]

def check_elbow(c,m,elbow_x,elbow_z,check):

    lin = m*elbow_x + c
    if elbow_z < lin:
        if check:
            return True
    else:
        return False


#Text Display
my_font = pygame.font.SysFont('Arial', 30,bold=True)
big_font = pygame.font.SysFont('Arial', 120,bold=True)

text_surface_front = my_font.render('Estimated Front View', False, (255, 255, 255))
text_surface_top = my_font.render('Estimated Top View', False, (255, 255, 255))
Warning_Text = big_font.render('ELLENBOGEN!!!', False, (200, 0, 0))


FPS = 0
block_sound=False
block_sound_counter=0
ELLENBOGEN_FLAG=False


#Main Loop
while True:
    try:
        #Limit to max. 60FPS (outdated)
        clock.tick(60)

        #get Images and Pose-Estimation-Points ffrom Estimator-Thread
        outputs1,outputs2,image1,image2 = Estimator.getKeypoints()

        # Reset the Estimated view Display
        view_xy = np.ones(shape=(480,640,3)) * 122
        view_yz = np.ones(shape=(480,640, 3)) * 122

        #Get only the Keypoints from the Network Output
        keypoints1 = outputs2['output_0'].numpy()
        keypoints2 = outputs1['output_0'].numpy()


        #preprocess the Keypoints to be in the range of values the Cameras were calibrated
        elbow1_x=int(round(keypoints1[0,0,8,0].copy()  * 320))
        elbow1_y = int(round(keypoints1[0, 0, 8, 1].copy() * 240))

        elbow2_x = int(round(keypoints2[0, 0, 8, 0].copy() * 320))
        elbow2_y = int(round(keypoints2[0, 0, 8, 1].copy() * 240))

        hand1_x = int(round(keypoints1[0,0,10,0].copy() * 320))
        hand1_y = int(round(keypoints1[0, 0, 10,1].copy() * 240))

        hand2_x = int(round(keypoints2[0, 0, 10, 0].copy() *320))
        hand2_y =int(round( keypoints2[0, 0, 10,1].copy() * 240))


        #Calculate Objectpoints
        elbow_3d = DLT(P0,P1,[elbow1_y,elbow1_x],[elbow2_y,elbow2_x])
        hand_3d = DLT(P0, P1, [hand1_y,hand1_x], [hand2_y,hand2_x])

        #Resize the Image for Display
        image1_arr = np.array(Image.open(io.BytesIO(image1)).resize((640, 480), 0))
        image2_arr = np.array(Image.open(io.BytesIO(image2)).resize((640, 480), 0))


        #preprocess calculated Object-Points to fit the Display and to be visible
        hand_3d_x = int(round((hand_3d[0]+100)*4))
        hand_3d_y = int(round((hand_3d[1]+40)*4))
        hand_3d_z = int(round((hand_3d[2]-80)*4))

        el_3d_x = int(round((elbow_3d[0]+100)*4))
        el_3d_y = int(round((elbow_3d[1]+40)*4))
        el_3d_z = int(round((elbow_3d[2]-80)*4))

        elbow_z =(elbow_3d[2])
        hand_z = (hand_3d[2])


        #Apply a small median-filter on the z-coordinate to filter High-Frequency Noise
        #Due to inaccuries and noise in the images, the calculated z-position is prone to high frequency noise
        #which could result in falsely alarming a player
        try :
            z_list_el.append(el_3d_z)
            z_list_hand.append(hand_3d_z)
            if len(z_list_el)>3:
                z_list_el.pop(0)
                z_list_hand.pop(0)
        finally:
            elbow_z = int(np.round(np.mean(z_list_el)))
            hand_z = int(np.round(np.mean(z_list_hand)))


        #Plot the Objectpoints in the Estimated View
        view_xy[ el_3d_y+100 - 10:el_3d_y+100 + 10, el_3d_x - 10:el_3d_x + 10, :] = [255,0,0]
        view_xy[hand_3d_y+100 - 10:hand_3d_y+100 + 10, hand_3d_x - 10:hand_3d_x + 10, :] = [0, 0, 255]

        view_yz[elbow_z-100 - 10:elbow_z-100 + 10, el_3d_x - 10:el_3d_x + 10, :] = [255, 0, 0]
        view_yz[hand_z-100 - 10:hand_z-100 + 10, hand_3d_x - 10:hand_3d_x + 10, :] = [0, 0, 255]

        view_yz[table_right[2]-100 - 10:table_right[2]-100 + 10, table_right[0] - 10:table_right[0] + 10, :] = [0, 255, 100]
        view_yz[table_left[2]-100 - 10:table_left[2]-100 + 10, table_left[0] - 10:table_left[0] + 10, :] = [0, 255, 100]

        # Plot the Edge of the Table
        try:

            view_yz[line_list[0],line_list[1]] = [0, 255, 100]
        except Exception as e:
            print(e)

        #Block the Sound, if the Sound is already playing
        if block_sound:
            block_sound_counter = block_sound_counter +1
            if block_sound_counter > 40:
                block_sound_counter = 0
                block_sound = False


        #preprocess image_arrays for displaying in pygame-window
        image1_arr = np.rot90(image1_arr)
        image2_arr = np.rot90(image2_arr)
        view_xy = np.rot90(view_xy)
        view_yz = np.rot90(view_yz)

        image1_arr = np.flip(image1_arr,axis=0)
        image2_arr = np.flip(image2_arr, axis=0)
        view_xy = np.flip(view_xy, axis=0)

        FPS_text = my_font.render('FPS: ' + str(int(FPS)), False, (255, 255, 255))

        screen.blit(pygame.surfarray.make_surface(image1_arr),(0,0))
        screen.blit(pygame.surfarray.make_surface(image2_arr),(640, 0))
        screen.blit(pygame.surfarray.make_surface(view_xy), (0, 480))
        screen.blit(pygame.surfarray.make_surface(view_yz), (640, 480))
        screen.blit(text_surface_front, (0, 480))
        screen.blit(text_surface_top, (640, 480))
        screen.blit(FPS_text, (1100, 900))

        #check if the player is overstepping the line
        if check_elbow(c,m,el_3d_x,elbow_z,startcheck):
            try:
                if block_sound == False:
                    block_sound = True
                    playsound('larm.wav',block=False)
                    print('ELLENBOGEN!!!')

            except Exception as e:
                print(e)

            screen.blit(Warning_Text, (200, 410))



        #finally update the display
        pygame.display.update()

        FPS = clock.get_fps()

    except Exception as e:
        print(e)


    #Possible Button Events
    #
    #
    # Right-Arrow: Reads in the right point of the Table
    #
    # Left-Arrow: Reads in the left point of the Table
    #
    # Up-Arrow: Enables the rule-checking and warning (can be annoying)
    # Down-Arrow: Disables the rule-checking and warning
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            image_stream.connection.close()
            image_stream.socket.close()
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                print('ellbogen: ', el_3d_x, el_3d_y, el_3d_z)  #
                print('hand: ', hand_3d_x, hand_3d_y, hand_3d_z)
                startcheck = False
              #  bodyparts_offset = bodyparts_3d
            if event.key == pygame.K_RIGHT:
                print('saved')
                table_left = [hand_3d_x, hand_3d_y, hand_3d_z]
                m,c = calculate_line_points(table_left,table_right)
                line_list = plot_table_line(c,m,table_left,table_right)
            if event.key == pygame.K_LEFT:
                print('saved')
                table_right = [hand_3d_x,hand_3d_y,hand_3d_z]
                m,c = calculate_line_points(table_right,table_left)
                line_list = plot_table_line(c, m, table_left, table_right)
            if event.key == pygame.K_UP:
                startcheck = True
