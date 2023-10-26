import io
import struct
import socket
import pygame
from PIL import Image
import threading


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
           # print('thread running')
            #print(self.connection.read(struct.calcsize('L')))
            package_size = struct.calcsize('<L')
            package = self.connection.read(package_size)
            self.image_len = struct.unpack('<L', package)[0]
            #print(self.image_len)
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
i=0
while True:
    try:

        clock.tick(60)

        image1,image2 = image_stream.get_Images()

        screenshot_1 = Image.open(io.BytesIO(image2))
        screenshot_2 = Image.open(io.BytesIO(image1))

        screen.blit(pygame.transform.scale(pygame.image.load(io.BytesIO(image2), "png"), (640, 480)), (0, 0))
        screen.blit(pygame.transform.scale(pygame.image.load(io.BytesIO(image1), "png"), (640, 480)), (640, 0))
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
                print('saved')
                screenshot_1.save(('Calibration_Pictures/Left/Calibration_l_%s.png' % i))
                screenshot_2.save(('Calibration_Pictures/Right/Calibration_r_%s.png' % i))
                screenshot_1.save(('Calibration_Pictures/Synched/Calibration_l_%s.png' % i))
                screenshot_2.save(('Calibration_Pictures/Synched/Calibration_r_%s.png' % i))
                i = i + 1