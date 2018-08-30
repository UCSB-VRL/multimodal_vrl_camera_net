"""
This file incorperates three detectors: HOG on RGB, HOG on themral, and depth segmentation with thermal heat checking.
The depth segmentation detection uses depth frame to get bounding boxes around big object in view and then uses thermal to check if there
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
            # print(np.mean(crop)) # print these for fine tuning
            # print(np.median(crop))
            # print(np.amin(crop))
            # print(np.amax(crop))
            if (np.mean(crop) > 15700 or np.amax(crop) - np.amin(crop) > 400):
                positive.append(rect)
            else:
                negative.append(rect)
    return positive, negative

def check_ir2(ir): #HOG detector on greyscale thermal data
    ir = ir - np.amin(ir)
    ir = np.uint8(ir.astype(float) * 255 / np.amax(ir) - 1)
    ir = cv2.cvtColor(ir.astype('uint8'), cv2.COLOR_GRAY2BGR)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(ir, winStride=(4,4), padding=(8,8), scale=1.05)
    rects = np.array([[x,y,x+w,y+h] for (x, y, w, h) in rects])
    full_rects = non_max_suppression(rects, probs=None, overlapThresh=0.65) #combine overlapping boxes
    if (type(full_rects) is np.ndarray):
        mylist = []
        for item in full_rects:
            mylist.append(item.tolist())
        return mylist
    return full_rects

def check_depth(depth): # segment depthmap
    d4d = np.uint8(depth.astype(float) * 255 / np.amax(depth) - 1) # make depth 8 bit
    cmap = cv2.Canny(d4d,100,200) # get edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)) # connect edges
    dilated = cv2.dilate(cmap, kernel)
    contoured, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE) # get contours from edge map
    rects = np.zeros(shape=(len(contours), 4))
    for i in range(len(contours)): # get coordinates for bounding boxes
        if len(contours[i]) > 50 and len(contours[i]) < 700:
            x, y, w, h = cv2.boundingRect(contours[i])
            rects[i] = [x, y, x+w, y+h]
    full_rects = non_max_suppression(rects, probs=None, overlapThresh=0.8) #combine overlapping boxes
    if (type(full_rects) is np.ndarray):
        mylist = []
        for item in full_rects:
            mylist.append(item.tolist())
        return mylist
    return full_rects

def check_rgb(rgb): #HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(rgb, winStride=(4,4), padding=(8,8), scale=1.05)
    rects = np.array([[x,y,x+w,y+h] for (x, y, w, h) in rects])
    full_rects = non_max_suppression(rects, probs=None, overlapThresh=0.65) #combine overlapping boxes
    if (type(full_rects) is np.ndarray):
        mylist = []
        for item in full_rects:
            mylist.append(item.tolist())
        return mylist
    return full_rects

def check_intersect(rgb, depth): #check to see how much depth bounding box intersects with rgb bounding box
    contained = 0.0
    total = (depth[2] - depth[0])*(depth[3] - depth[1])
    for x in range(depth[0], depth[2]):
        for y in range(depth[1], depth[3]):
            if (x > rgb[0] and x < rgb[2] and y > rgb[1] and y < rgb[3]):
                contained = contained + 1
    return contained / total

def human_detector(rgb, depth, ir): #takes raw images without homography applied
    strong_rects = []
    medium_rects = []
    homog = get_pos()
    depthm = np.mean(depth[70:170,110:210])
    rgb = homog.rgb_conv(rgb, depthm) # convert rgb with homography
    depth = homog.rgb_conv(depth, depthm) # convert depth with homography
    rgb_rects = check_rgb(rgb)
    depth_rects = check_depth(depth)
    pos_depth, neg_depth = check_ir(depth_rects, ir)
    ir_rects = check_ir2(ir)
    for rgb_rect in rgb_rects:
        for ir_rect in ir_rects:
            if (check_intersect(rgb_rect, ir_rect) > 0.7): #70% contained
                if (not any((rgb_rect == x).all() for x in strong_rects)): #only append if not in there
                    strong_rects.append(rgb_rect) #add to strong if in rgb and ir
                rgb_rects.remove(rgb_rect) # combine boxes
                ir_rects.remove(ir_rect)
    depth_remove = []
    rgb_remove = []
    ir_remove = []
    for depth_rect in pos_depth:
        for strong_rect in strong_rects:
            if (check_intersect(strong_rect, depth_rect) > 0.5): #50% contained
                depth_remove.append(depth_rect) # combine boxes
        for rgb_rect in rgb_rects:
            if (check_intersect(rgb_rect, depth_rect) > 0.5): #50% contained
                if (not any((rgb_rect == x) for x in medium_rects)): #only append if not in there
                    medium_rects.append(rgb_rect) # add to medium if rgb and depth detect
                depth_remove.append(depth_rect)
                rgb_remove.append(rgb_rect)
        for ir_rect in ir_rects:
            if (check_intersect(ir_rect, depth_rect) > 0.5): #50% contained
                if (not any((ir_rect == x) for x in medium_rects)): #only append if not in there
                    medium_rects.append(ir_rect) # add to medium if ir and depth setect
                depth_remove.append(depth_rect) # combine boxes
                ir_remove.append(ir_rect) # combine boxes
    # remove boxes from remove list
    for item in ir_remove:
        try:
            ir_rects.remove(item)
        except:
            pass
    for item in rgb_remove:
        try:
            rgb_rects.remove(item)
        except:
            pass
    for item in depth_remove:
        try:
            pos_depth.remove(item)
        except:
            pass

    for rect in strong_rects: #green for strong prediciton (HOG postive from rgb and thermal)
        cv2.rectangle(rgb, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)
    for rect in medium_rects: #yellow for medium predicition (depth with either rgb or thermal HOG)
        cv2.rectangle(rgb, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 1)
    for rect in ir_rects: #orange for low prediction (just thermal HOG)
        cv2.rectangle(rgb, (rect[0], rect[1]), (rect[2], rect[3]), (0, 100, 255), 1)
    for rect in rgb_rects: #orange for low prediction (just rgb HOG)
        cv2.rectangle(rgb, (rect[0], rect[1]), (rect[2], rect[3]), (0, 100, 255), 1)
    for rect in pos_depth: #red for poor prediciton (just depth/ir detection)
        cv2.rectangle(rgb, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 1)

    # cv2.imshow("boxes", rgb)
    # cv2.waitKey(0)

    return strong_rects, medium_rects, rgb_rects, ir_rects, pos_depth, rgb
    

##### USED FOR TESTING #####

# curdir = "/home/carlos/vrlserver/videos/raw/cam1/recording_1/"
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

# fullpath = "/home/carlos/vrlserver/videos/raw/cam1/recording_1/1/"
# item = 14
# depth = np.load(fullpath + "/depth_full_vid/depth_frame_" + str(item) + ".npy")
# ir = np.load(fullpath + "/ir_full_vid/ir_frame_" + str(item) + ".npy")
# rgb = np.load(fullpath + "/rgb_full_vid/rgb_frame_" + str(item) + ".npy")
# human_detector(rgb, depth, ir)
