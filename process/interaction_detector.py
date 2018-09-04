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
thresh = 0.35
hier_thresh = 0.7
pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)

def interaction_detector(rgb,depth,ir):
    persons = []
    objects = []
    homog = get_pos()
    depthm = np.mean(depth[70:170,110:210])
    rgbhomog = homog.rgb_conv(rgb, depthm) # convert rgb with homography
    rgbnew = rgb.transpose(2,0,1)
    c, h, w = rgbnew.shape[0], rgbnew.shape[1], rgbnew.shape[2]
    data = rgbnew.ravel()/255.0
    data = np.ascontiguousarray(data, dtype=np.float32)
    outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh) # output dictionary of objects classified

    for output in outputs:
        valid, out_homog = correct_bounds(output, depthm) # check to see if object is in homography view
        if valid:
            cv2.rectangle(rgbhomog, (out_homog['left_h'], out_homog['top_h']), (out_homog['right_h'], out_homog['bottom_h']), (255, 0, 0), 1)
            if output['class'] == 'person':
                persons.append(out_homog) #create set of humans
            else:
                objects.append(out_homog) #create set of objects
    
    if (len(persons) > 0 and len(objects) > 0):
        interacting(persons, objects, ir, depth) #if there is atleast 1 object and 1 person see if they are interacting
    if (len(persons) == 0):
        print("No person detected")
    if (len(objects) == 0):
        print("No object detected")

    # cv2.imshow("boxeshomog", rgbhomog)
    # cv2.waitKey(0)
        
    return rgbhomog

        
def correct_bounds(output, depthm): # checks to see if part of the object is in view of the homography
    homog = get_pos()
    valid = False
    output['left_h'], output['top_h'] = homog.rgb_to_ir(output['left'], output['top'], depthm)
    output['right_h'], output['bottom_h'] = homog.rgb_to_ir(output['right'], output['bottom'], depthm)
    if (not(output['right_h'] < 0 or output['left_h'] > 156) and not(output['bottom_h'] < 0 or output['top_h'] > 206)):
        valid = True
    return valid, output

def nearby_box(person, my_object): #checks to see if two bounding boxes are close or overlapping
    extra_dist = 10 # max pixel distance between bounding boxes
    if (not(my_object['right'] < person['left'] - extra_dist or my_object['left'] > person['right'] + extra_dist) and not(my_object['bottom'] < person['top'] - extra_dist or my_object['top'] > person['bottom'] + extra_dist)):
        return True
    return False

def interacting(persons, objects, ir, depth): #check if there is interactions through ir
    for person in persons:
        for my_object in objects:
            print("person may be interacting with:", my_object['class'])
            # check how close bounding boxes are
            depth_diff = abs(np.mean(depth[my_object['top']:my_object['bottom'], my_object['left']:my_object['right']]) - np.mean(depth[person['top']:person['bottom'], person['left']:person['right']]))
            if (nearby_box(person, my_object) and depth_diff < 50): #check for heat interaction
                heat_diff = np.mean(ir[person['top']:person['bottom'], person['left']:person['right']]) - np.amin(ir[person['top']:person['bottom'], person['left']:person['right']])
                if(heat_diff > 80):
                    print("person interacting with:", my_object['class'])
                else:
                   print("person not interacting with:", my_object['class']) 

def end_classification():
    pyyolo.cleanup()


####### USED FOR TESTING ########
# fullpath = "/home/carlos/vrlserver/videos/raw/cam1/recording_6"
# for item in range(22,23):
#     ir = np.load(fullpath + "/ir_full_vid/ir_frame_" + str(item) + ".npy")
#     depth = np.load(fullpath + "/depth_full_vid/depth_frame_" + str(item) + ".npy")
#     rgb = np.load(fullpath + "/rgb_full_vid/rgb_frame_" + str(item) + ".npy")
#     interaction_detector(rgb,depth,ir)
# end_classification()