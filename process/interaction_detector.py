"""
Uses RGB object detection through yolo and thermal information to see if human is 
interacting with object."
"""

import numpy as np
import cv2
import os
import sys
sys.path.insert(0, '../src')
from use_homographies import get_pos
import pyyolo

# darknet values
darknet_path = '../shells/pyyolo/darknet'
datacfg = 'cfg/coco.data'
cfgfile = '../tiny-yolo.cfg'
weightfile = '../tiny-yolo.weights'
thresh = 0.45
hier_thresh = 0.5
pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)

def runYOLO(rgb,depth):
    homog = get_pos()
    depthm = np.mean(depth[70:170,110:210])
    rgbhomog = homog.rgb_conv(rgb, depthm) # convert rgb with homography
    rgbnew = rgb.transpose(2,0,1)
    c, h, w = rgbnew.shape[0], rgbnew.shape[1], rgbnew.shape[2]
    data = rgbnew.ravel()/255.0
    data = np.ascontiguousarray(data, dtype=np.float32)
    outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)
    for output in outputs:
        valid, first, second = correct_bounds(output)
        if valid:
            cv2.rectangle(rgbhomog, (first[0], first[1]), (first[0], first[1]), (255, 0, 0), 1)
    cv2.imshow("boxes", rgbhomog)
    cv2.waitKey(0)
        
def correct_bounds(output):
    print(homog.rgb_to_ir(output['left'], output['top'], depthm))
    print(homog.rgb_to_ir(output['right'], output['bottom'], depthm))
    return 

def get_homog(rgb, depth):
    homog = get_pos()
    depthm = np.mean(depth[70:170,110:210])
    rgb = homog.rgb_conv(rgb, depthm) # convert rgb with homography
    return(rgb)

def interacting(): #check if there is interactions through ir
    print("True")

def endSess():
    pyyolo.cleanup()

fullpath = "/home/carlos/vrlserver/videos/Cam 1/open door/13"
for item in range(16,21):
    ir = np.load(fullpath + "/ir_full_vid/ir_frame_" + str(item) + ".npy")
    depth = np.load(fullpath + "/depth_full_vid/depth_frame_" + str(item) + ".npy")
    rgb = np.load(fullpath + "/rgb_full_vid/rgb_frame_" + str(item) + ".npy")
    runYOLO(rgb,depth)
    # get_homog(rgb, depth)