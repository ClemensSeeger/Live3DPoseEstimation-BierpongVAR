# Image Transfer - As The Controller Device
#
# This script is meant to talk to the "image_transfer_jpg_streaming_as_the_remote_device_for_your_computer.py" on the OpenMV Cam.
#
# This script shows off how to transfer the frame buffer to your computer as a jpeg image.

import io, pygame, rpc, serial, serial.tools.list_ports, socket, sys
import struct
import threading
import time
# Fix Python 2.x.
try: input = raw_input
except NameError: pass

# The RPC library above is installed on your OpenMV Cam and provides mutliple classes for
# allowing your OpenMV Cam to control over USB or WIFI.

##############################################################
# Choose the interface you wish to control an OpenMV Cam over.
##############################################################

# Uncomment the below lines to setup your OpenMV Cam for controlling over a USB VCP.
#
# * port - Serial Port Name.
#
print("\nAvailable Ports:\n")
for port, desc, hwid in serial.tools.list_ports.comports():
    print("{} : {} [{}]".format(port, desc, hwid))
sys.stdout.flush()

class CamStream:
    def __init__(self,src):
        self.capture = rpc.rpc_usb_vcp_master(src)
        self.thread = threading.Thread(target=self.update,daemon=True)
        self.thread.start()
    def update(self):
        while True:
            self.result = self.capture.call("jpeg_image_stream", "sensor.RGB565,sensor.QQVGA")
            if (self.result) is not None:
                try: self.capture._stream_put_bytes(self.capture._set_packet(0xEDF6, struct.pack("<I", 8)), 1000)
                except OSError: return
                tx_lfsr = 255
                while True:
                    packet = self.capture._stream_get_bytes(bytearray(8), 1000)
                    if packet is None: return
                    magic = packet[0] | (packet[1] << 8)
                    crc = packet[-2] | (packet[-1] << 8)
                    if magic != 0x542E and crc != self.capture.__crc_16(packet, len(packet) - 2): return
                    self.data = self.capture._stream_get_bytes(bytearray(struct.unpack("<I", packet[2:-2])[0]), 5000)

                    try: self.capture._stream_put_bytes(struct.pack("<B", tx_lfsr), 1000)
                    except OSError: return
                    tx_lfsr = (tx_lfsr >> 1) ^ (0xB8 if tx_lfsr & 1 else 0x00)
    #def placeholder_cb(data):
    def getFrame(self):

        return self.data



cam_1 = CamStream('/dev/ttyACM0')
cam_2 = CamStream('/dev/ttyACM1')


server_socket = socket.socket()
server_socket.connect(('192.168.1.138',8000))

connection = server_socket.makefile('wb')



pygame.init()
screen_w = 640
screen_h = 480
try:
    screen = pygame.display.set_mode((screen_w, screen_h), flags=pygame.RESIZABLE)
except TypeError:
    screen = pygame.display.set_mode((screen_w, screen_h))
pygame.display.set_caption("Frame Buffer")
clock = pygame.time.Clock()





while(True):
    sys.stdout.flush()
    try:

        frame1 = cam_1.getFrame()
        frame2 = cam_2.getFrame()

        if frame1 is not None and frame2 is not None:


            clock.tick(60)
            connection.write(struct.pack('<L',len(frame1)))
            connection.flush()
            connection.write(frame1)
      
            connection.write(struct.pack('<L',len(frame2)))
            connection.flush()
            connection.write(frame2)

    except Exception as e:
        print(e)
        
    print(clock.get_fps())
        
    
        
        
        
        
        
 