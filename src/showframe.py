"""
Use this script to view rgb, depth and thermal frames in the current directory.
"""
import numpy as np
import cv2
import os
from use_homographies import get_pos

def get_8bit_frame(frame): #represnt thermal image in rgb space for other window
    frame = frame - np.amin(frame)
    ir_frame = np.uint8(frame.astype(float) * 255 / np.amax(frame) - 1)
    ir_frame = cv2.cvtColor(ir_frame.astype('uint8'), cv2.COLOR_GRAY2BGR)
    return ir_frame

def get_depth_frame(frame):
    depth_frame = np.uint8(frame.astype(float) * 255 / np.amax(frame) - 1)
    depth_frame = 255 - cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2RGB)
    return depth_frame

curdir = os.getcwd()
while (1):
    print("Type frame number to view")
    curframe = raw_input()
    try: #load and display frame
        rgb_frame = np.load(curdir+"/rgb_full_vid/rgb_frame_"+curframe+".npy")
        depth = np.load(curdir+"/depth_full_vid/depth_frame_"+curframe+".npy")
        depth_frame = get_depth_frame(depth)
        ir = np.load(curdir+"/ir_full_vid/ir_frame_"+curframe+".npy")
        ir_frame = get_8bit_frame(ir)
        #resize ir to hstack with rgb and depth
        ir_resized = np.zeros((240,156))
        ir_resized = cv2.cvtColor(ir_resized.astype('uint8'), cv2.COLOR_GRAY2BGR)
        ir_resized[34:240,:] = ir_frame
        disp = np.hstack((rgb_frame, depth_frame, ir_resized))

        homog = get_pos()
        depthm = np.mean(depth[70:170,110:210]) #avg depth in middle of frame
        disp_irhomog = np.hstack((homog.rgb_conv(rgb_frame,depthm), homog.rgb_conv(depth_frame,depthm), ir_frame))
        
        cv2.imshow('frame', disp) #without homography
        cv2.imshow('frame_homog', disp_irhomog) #with homography
        cv2.waitKey(500)

    except: #print out valid frames
        files = os.listdir(curdir+"/rgb_full_vid/")
        for i in range(len(files)):
            files[i] = files[i].split('_')[2].strip(".npy")
        files.sort(key=int)
        print("Invalid frame number. Valid frames are:", files)
