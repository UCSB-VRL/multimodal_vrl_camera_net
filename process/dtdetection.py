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
from imutils.object_detection import non_max_suppression

def check_ir(rects, frame): #if there is a strong heat signature in the depth region then return a positive
    positive = []
    negative = []
    for rect in rects:
        if (not np.array_equiv(rect, [0, 0, 0, 0])):
            crop = frame[rect[1]:rect[3],rect[0]:rect[2]]
            flat = crop.flatten()
            # print(np.mean(crop))
            # print(np.median(crop))
            # print(np.amin(crop))
            # print(np.amax(crop))
            if (np.mean(crop) > 15600 or np.amax(crop) - np.amin(crop) > 400):
                positive.append(rect)
            else:
                negative.append(rect)
    return np.array(positive), np.array(negative)

def check_depth(depth): # segment depthmap
    d4d = np.uint8(depth.astype(float) * 255 / 2**12 - 1) # make depth 8 bit
    d4d = d4d/32 # round to 3 bit
    d4d = d4d*32 # make rounded number in 8 bit space
    cmap = cv2.Canny(d4d,0,250) # get edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) # connect edges
    dilated = cv2.dilate(cmap, kernel)
    contoured, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE) # get contours from edge map
    rects = np.zeros(shape=(len(contours), 4))
    for i in range(len(contours)): # get coordinates for bounding boxes
        if len(contours[i]) > 50 and len(contours[i]) < 800:
            x, y, w, h = cv2.boundingRect(contours[i])
            rects[i] = [x, y, x+w, y+h]
    full_rects = non_max_suppression(rects, probs=None, overlapThresh=.4)
    return full_rects

def check_rgb(rgb): #HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(rgb, winStride=(4,4), padding=(8,8), scale=1.05)
    rects = np.array([[x,y,x+w,y+h] for (x, y, w, h) in rects])
    full_rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    return full_rects

def human_detector(rgb, depth, ir): #takes raw images without homography applied
    homog = get_pos()
    depthm = np.mean(depth[70:170,110:210])
    rgb = homog.rgb_conv(rgb, depthm) # convert rgb with homography
    depth = homog.rgb_conv(depth, depthm) # convert depth with homography
    rgb_rects = check_rgb(rgb)
    depth_rects = check_depth(depth)
    pos_depth, neg_depth = check_ir(depth_rects, ir)

    for rect in rgb_rects:
        cv2.rectangle(rgb, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 1) #blue for rgb human in it
    for rect in pos_depth:
        cv2.rectangle(rgb, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1) #green for depth&ir human in it
    for rect in neg_depth:
        cv2.rectangle(rgb, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 1) #red for no ir human in it

    if (len(rgb_rects) > 0 and pos_depth.shape[0] > 0):
        combined = np.append(rgb_rects, pos_depth, axis=0)
        strong_pos = non_max_suppression(combined, probs=None, overlapThresh=0.4)
        
                    
        for rect in strong_pos:
            cv2.rectangle(rgb, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 0), 1) #red for no ir human in it

    cv2.imshow("boxes", rgb)
    cv2.waitKey(0)
    

# curdir = "/home/carlos/vrlserver/videos/Cam 1/open door/"
# folders = os.listdir(curdir)
# folders = [x for x in folders if " " not in x] #find only folders that are a number
# folders.sort(key=int)
# for folder in folders:
#     print(folder)
#     files = os.listdir(curdir + folder + "/rgb_full_vid/")
#     for i in range(len(files)):
#         files[i] = files[i].split('_')[2].strip(".npy")
#     files.sort(key=int)
#     for item in files:
#         depth = np.load(curdir + folder + "/depth_full_vid/depth_frame_" + str(item) + ".npy")
#         ir = np.load(curdir + folder + "/ir_full_vid/ir_frame_" + str(item) + ".npy")
#         rgb = np.load(curdir + folder + "/rgb_full_vid/rgb_frame_" + str(item) + ".npy")
#         human_detector(rgb, depth, ir)

fullpath = "/home/carlos/vrlserver/videos/Cam 1/open door/13"
item = 17
depth = np.load(fullpath + "/depth_full_vid/depth_frame_" + str(item) + ".npy")
ir = np.load(fullpath + "/ir_full_vid/ir_frame_" + str(item) + ".npy")
rgb = np.load(fullpath + "/rgb_full_vid/rgb_frame_" + str(item) + ".npy")
human_detector(rgb, depth, ir)