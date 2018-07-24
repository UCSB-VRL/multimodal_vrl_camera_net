import numpy as np
import cv2
import os

def start():
    global curdir
    print("Type directory containing frame folders or nothing to use current working directory.")
    curdir = raw_input()
    if (curdir == ""):
        curdir = os.getcwd()

def get_8bit_frame(frame):
    output = frame >> 2
    output[output >= 4096] = 4096 - 1
    resized = np.zeros((240,156))
    resized[34:240,:] = output
    resized = cv2.cvtColor(resized.astype('uint8'), cv2.COLOR_GRAY2BGR)
    return resized

start()
while (1):
    print("Type frame number to view or n to choose new directory")
    curframe = raw_input()
    if (curframe == "n"):
        start()
    else:
        rgb_frame = np.load(curdir+"/rgb_full_vid/rgb_frame_"+curframe+".npy")
        depth = np.load(curdir+"/depth_full_vid/depth_frame_"+curframe+".npy")
        ir = np.load(curdir+"/ir_full_vid/ir_frame_"+curframe+".npy")

        ir_frame = get_8bit_frame(ir)
        depth_frame = np.uint8(depth.astype(float) * 255 / 2**12 - 1)
        depth_frame = 255 - cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2RGB)

        disp = np.hstack((rgb_frame, depth_frame, ir_frame))
        cv2.imshow('frame', disp)
        cv2.waitKey(500)
