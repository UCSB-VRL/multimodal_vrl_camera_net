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
darknet_path = './darknet'
datacfg = 'cfg/coco.data'
cfgfile = 'cfg/tiny-yolo.cfg'
weightfile = '../tiny-yolo.weights'
thresh = 0.45
hier_thresh = 0.5
pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)

def runYOLO(rgb):
    rgb = rgb.transpose(2,0,1)
	c, h, w = rgb.shape[0], rgb.shape[1], rgb.shape[2]
	data = rgb.ravel()/255.0
	data = np.ascontiguousarray(data, dtype=np.float32)
	outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)	
	for output in outputs:
		print(output)

def endSess():
    pyyolo.cleanup()