# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:55:46 2017

@author: julian
"""
import numpy as np
import cv2

class get_pos():
    def get_homography(self, distance, space):
        for i in range(self.num_homographies):
            if (self.dist[i] >= distance or i == 4): # homography distance >= frame distance
                if space == 'rgb':
                    return self.h_rgb[:, :, i]
                elif space == 'ir':
                    return self.h_ir[:, :, i]

    def rgb_to_ir(self, x, y, distance):
        pos = np.array((x, y, 1))
        homography = self.get_homography(distance, 'rgb')
        new_pos = homography.dot(pos)
        out = [new_pos[0], new_pos[1]] / new_pos[2]
        return out

    def ir_to_rgb(self, x, y, distance):
        pos = np.array((x, y, 1))
        homography = self.get_homography(distance, 'ir')
        new_pos = homography.dot(pos)
        out = [new_pos[0], new_pos[1]] / new_pos[2]
        return out

    def rgb_conv(self, img, distance):
        homography = self.get_homography(distance, 'rgb')
        return cv2.warpPerspective(img, homography, (156,206), flags = cv2.INTER_NEAREST) #shape of ir

    def ir_conv(self, img, distance):
        homography = self.get_homography(distance, 'ir')
        return cv2.warpPerspective(img, homography, (320,240), flags = cv2.INTER_NEAREST) #shape of rgb/depth - nearest interpolation used to get rid of noise

    def __init__(self, num_homographies=5):
        self.num_homographies = num_homographies
        self.h_rgb = np.zeros((3, 3, self.num_homographies))
        self.h_ir = np.zeros((3, 3, self.num_homographies))
        self.dist = np.zeros((self.num_homographies, 1))
        for i in range(1, self.num_homographies + 1):
            temp1 = np.loadtxt('/home/carlos/Documents/multimodal_vrl_camera_net/src/Homographies/Hmatrix_rgb_to_ir_' + str(i) + '.out')
            temp2 = np.loadtxt('/home/carlos/Documents/multimodal_vrl_camera_net/src/Homographies/Hinvmatrix_ir_to_rgb_' + str(i) + '.out')
            self.h_rgb[:, :, i - 1] = temp1[0:3, 0:3]
            self.h_ir[:, :, i - 1] = temp2[0:3, 0:3]
            self.dist[i - 1] = temp1[3, 0]

# THE FOLLOWING CODE IS FOR HOMOGRAPHY TESTING
# def get_8bit_frame(frame): #represnt thermal image in rgb space
#     output = frame >> 2
#     output[output >= 4096] = 4096 - 1
#     output = cv2.cvtColor(output.astype('uint8'), cv2.COLOR_GRAY2BGR)
#     return output

# curframe = "7"
# curdir = "/home/carlos/vrlserver/videos/drink water/1"

# rgb = np.load(curdir+"/rgb_full_vid/rgb_frame_"+curframe+".npy")
# depth = np.load(curdir+"/depth_full_vid/depth_frame_"+curframe+".npy")
# ir = np.load(curdir+"/ir_full_vid/ir_frame_"+curframe+".npy")

# # Get depth of subject - using closest depth from middle of img
# depthm = np.mean(depth[70:170,110:210])

# test = get_pos()
# cv2.imshow('irnew', get_8bit_frame(test.ir_conv(ir,depthm)))
# cv2.imshow('rgbnew', test.rgb_conv(rgb,depthm))
# cv2.imshow('rgbold', rgb)
# cv2.imshow('irold', get_8bit_frame(ir))
# cv2.waitKey(0)
