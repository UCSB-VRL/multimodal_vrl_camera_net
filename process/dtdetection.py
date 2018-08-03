"""
Uses depth frame to get bounding boxes around big object is view and then uses thermal to check if there
is a significant heat signature to predict if human is in frame or not.
"""

import numpy as np
import cv2
import os
import sys
sys.path.insert(0, '../src')
from use_homographies import get_pos

def check_ir(x, y, w, h, frame): #if there is a strong heat signature in the depth region then return a positive
    crop = frame[y:y+h,x:x+w]
    flat = crop.flatten()
    # print(np.mean(crop))
    # print(np.median(crop))
    # print(np.amin(crop))
    # print(np.amax(crop))
    if (np.mean(crop) > 15600 or np.amax(crop) - np.amin(crop) > 400):
        print("person detected (d&t)")
        return 1
    else:
        return 0

def check_depth(depth, ir, rgb):
    homog = get_pos()
    depthm = np.mean(depth[70:170,110:210])
    frame = homog.rgb_conv(depth, depthm) # convert depth with homography
    rgb = homog.rgb_conv(rgb, depthm)
    d4d = np.uint8(frame.astype(float) * 255 / 2**12 - 1) # make depth 8 bit
    d4d = d4d/32 # round to 3 bit
    d4d = d4d*32 # make rounded number in 8 bit space
    cmap = cv2.Canny(d4d,0,250) # get edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) # connect edges
    dilated = cv2.dilate(cmap, kernel)
    contoured, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE) # get contours from edge map
    im = cv2.cvtColor(d4d, cv2.COLOR_GRAY2BGR)
    for i in range(len(contours)): # draw bounding boxes and check heat signatures to confirm if human
        if len(contours[i]) > 20 and len(contours[i]) < 800:
            x, y, w, h = cv2.boundingRect(contours[i])
            if check_ir(x,y,w,h,ir):
                cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 1) #green for human in it
            else:
                cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 0, 255), 1) #red for not
    cv2.imshow("boxes", rgb)
    cv2.waitKey(0)

curdir = "/home/carlos/vrlserver/videos/Cam 1/open door/"
folders = os.listdir(curdir)
folders = [x for x in folders if " " not in x]
folders.sort(key=int)
for folder in folders:
    print(folder)
    files = os.listdir(curdir + folder + "/rgb_full_vid/")
    for i in range(len(files)):
        files[i] = files[i].split('_')[2].strip(".npy")
    files.sort(key=int)
    for item in files:
        depth = np.load(curdir + folder + "/depth_full_vid/depth_frame_" + str(item) + ".npy")
        ir = np.load(curdir + folder + "/ir_full_vid/ir_frame_" + str(item) + ".npy")
        rgb = np.load(curdir + folder + "/rgb_full_vid/rgb_frame_" + str(item) + ".npy")
        check_depth(depth, ir, rgb)