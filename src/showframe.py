"""
Use this script to view rgb, depth and thermal frames in the current directory.
"""
import numpy as np
import cv2
import os
from use_homographies import get_pos

def get_8bit_frame(frame): #represnt thermal image in rgb space
    output = frame >> 2
    output[output >= 4096] = 4096 - 1
    resized = np.zeros((240,156))
    resized[34:240,:] = output
    resized = cv2.cvtColor(resized.astype('uint8'), cv2.COLOR_GRAY2BGR)
    return resized

def get_8bit_frame_homog(frame): #represnt thermal image in rgb space
    output = frame >> 2
    output[output >= 4096] = 4096 - 1
    output = cv2.cvtColor(output.astype('uint8'), cv2.COLOR_GRAY2BGR)
    return output

curdir = os.getcwd()
while (1):
    print("Type frame number to view or n to choose new directory")
    curframe = raw_input()
    try: #load and display frame
        rgb_frame = np.load(curdir+"/rgb_full_vid/rgb_frame_"+curframe+".npy")
        depth = np.load(curdir+"/depth_full_vid/depth_frame_"+curframe+".npy")
        ir = np.load(curdir+"/ir_full_vid/ir_frame_"+curframe+".npy")
        ir_frame = get_8bit_frame(ir)
        depth_frame = np.uint8(depth.astype(float) * 255 / 2**12 - 1)
        depth_frame = 255 - cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2RGB)
        disp = np.hstack((rgb_frame, depth_frame, ir_frame))

        homog = get_pos()
        depthm = np.mean(depth[70:170,110:210])
        disp_irhomog = np.hstack((homog.rgb_conv(rgb_frame,depthm), homog.rgb_conv(depth_frame,depthm), get_8bit_frame_homog(ir)))
        
        cv2.imshow('frame', disp)
        cv2.imshow('frame_homog', disp_irhomog)
        cv2.waitKey(500)

    except: #print out valid frames
        files = os.listdir(curdir+"/rgb_full_vid/")
        for i in range(len(files)):
            files[i] = files[i].split('_')[2].strip(".npy")
        files.sort(key=int)
        print("Invalid frame number. Valid frames are:", files)