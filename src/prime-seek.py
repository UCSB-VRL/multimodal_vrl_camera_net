"""
Created on Mon Jul  3 09:54:53 2017

@author: Julian

Use to record with the primesense camera RGB and depth cameras and the seek thermal camera
"""
import numpy as np
import cv2
import os
from primesense import openni2  # , nite2
from primesense import _openni2 as c_api
from seek_camera import thermal_camera
from use_homographies import get_pos
import sys
import time
sys.path.append('../process')

# Device number
devN = 1

#############################################################################
# set video saving location
#base_video_location = '/home/carlos/Videos/' #locally
base_video_location = '/home/carlos/vrlserver/videos/raw/cam' + str(devN) + '/' #intern server

# set-up primesense camera
dist = "/home/carlos/Install/kinect/OpenNI-Linux-Arm-2.2/Redist"
# Initialize openni and check
openni2.initialize(dist)
if (openni2.is_initialized()):
    print "openNI2 initialized"
else:
    print "openNI2 not initialized"
# Register the device
prime = openni2.Device.open_any()
# Create the streams
rgb_stream = prime.create_color_stream()
depth_stream = prime.create_depth_stream()
# Configure the depth_stream -- changes automatically based on bus speed
# print 'Depth video mode info', depth_stream.get_video_mode() # Checks depth video configuration
depth_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=320, resolutionY=240, fps=30))
# Start the streams
rgb_stream.start()
depth_stream.start()
# Synchronize the streams
prime.set_depth_color_sync_enabled(True)
# IMPORTANT: ALIGN DEPTH2RGB (depth wrapped to match rgb stream)
prime.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)


def get_rgb():
    """
    Returns numpy 3L ndarray to represent the rgb image.
    """
    bgr = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),
                        dtype=np.uint8).reshape(240, 320, 3)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def get_depth():
    """
    Returns numpy ndarrays representing the raw and ranged depth images.
    Outputs:
        dmap:= distancemap in mm, 1L ndarray, dtype=uint16, min=0, max=2**12-1
        d4d := depth for dislay, 3L ndarray, dtype=uint8, min=0, max=255    
    Note1: 
        fromstring is faster than asarray or frombuffer
    Note2:     
        .reshape(120,160) #smaller image for faster response 
                OMAP/ARM default video configuration
        .reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
                Requires .set_video_mode
    """
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(),
                         dtype=np.uint16).reshape(240, 320)  # Works & It's FAST
    # Correct the range. Depth images are 12bits
    d4d = np.uint8(dmap.astype(float) * 255 / np.amax(dmap) - 1)
    d4d = 255 - cv2.cvtColor(d4d, cv2.COLOR_GRAY2RGB)
    return dmap, d4d

def makedir(): #increments recording number and creates all the necessary folders
    global rgb_vid, ir_vid, depth_vid, ir_name, depth_name, rgb_name, recording, video_location
    recording = 1
    video_location = base_video_location
    while os.path.exists(video_location+'recording_'+str(recording)+'/'):
        recording += 1
    recording = str(recording)
    oldmask = os.umask(000)
    os.makedirs(video_location+'recording_'+recording+'/', 0777)
    os.umask(oldmask)
    video_location =  video_location+'recording_'+recording+'/'
    rgb_vid = cv2.VideoWriter(video_location + 'rgb_vid.avi', fourcc, fps, (rgb_w, rgb_h), 1)
    ir_vid = cv2.VideoWriter(video_location + 'ir_vid.avi', fourcc, fps, (ir_w, ir_h), 1)
    depth_vid = cv2.VideoWriter(video_location + 'depth_vid.avi', fourcc, fps, (depth_w, depth_h), 1)

    os.makedirs(video_location+'ir_full_vid')
    os.makedirs(video_location+'depth_full_vid')
    os.makedirs(video_location+'rgb_full_vid')
    ir_name = video_location+'ir_full_vid/ir_frame_'
    depth_name = video_location+'depth_full_vid/depth_frame_'
    rgb_name = video_location+'rgb_full_vid/rgb_frame_'

# setup thermal camera
therm = thermal_camera()
# get frames
rgb_frame = get_rgb()
dmap, depth_frame = get_depth()
ir_frame = therm.get_frame()
# setup needed information for displaying non homography image
rgb_h, rgb_w, channels = rgb_frame.shape
ir_h, ir_w = ir_frame.shape
depth_h, depth_w, depth_channels = depth_frame.shape
ir_place = np.zeros((rgb_h, ir_w, channels), dtype='uint8')
depth_place = np.zeros((depth_h, depth_w, channels), dtype='uint8')
place_ir = rgb_h / 2 - ir_h / 2
place_depth = rgb_h / 2 - depth_h / 2

# ==============================================================================
# THE CODECS
# ==============================================================================
if cv2.__version__ == '3.1.0':
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
else:
    fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')

print ("Press 'esc' to terminate, 'r' to record, 's' to stop recording")
print ("'0' for no homography, '1' for rgb/d perspective, '2' for thermal perspective")
print ("'h' for human detector, 'i' for interaction detector")

f = 0   # frame counter
fr = 0 #frame rate counter
done = False
rec = False
start = time.time()
full_start = time.time()
frame_rate=""
# ==============================================================================
# Real time inference
# ==============================================================================
human_detection = False
interaction_detection = False

# 0 = no homography, 1 = RGB/d perspective,2 = IR perspective
homography_setting = 0 #defaults to 2 if running inference

while not done:
    if (human_detection):
        import human_detector
    if (interaction_detection):
        import interaction_detector

    if (interaction_detection):
        k = cv2.waitKey(60) & 255
    else:
        k = cv2.waitKey(1) & 255

    # capture frames
    rgb_frame = get_rgb()
    full_ir = therm.get_frame()
    full_depth, depth_frame = get_depth()
    rgb_frame = cv2.flip(rgb_frame,1)
    full_ir = cv2.flip(full_ir,1)
    full_depth = cv2.flip(full_depth,1)
    depth_frame = cv2.flip(depth_frame,1)
    ir_frame = therm.get_8bit_frame(full_ir)

    # homographies
    if (homography_setting == 2 or human_detection or interaction_detection):
        homog = get_pos()
        depthm = np.mean(dmap[70:170,110:210])
        ir_place = ir_frame #IR view
        depth_place = homog.rgb_conv(depth_frame,depthm) #IR view
        rgb_place = homog.rgb_conv(rgb_frame,depthm) #IR view
    elif homography_setting == 1: #RGB/D view
        homog = get_pos()
        depthm = np.mean(dmap[70:170,110:210])
        ir_place = homog.ir_conv(ir_frame,depthm) 
        rgb_place = rgb_frame
        depth_place = depth_frame
    else:
        ir_place = np.zeros((rgb_h, ir_w, channels), dtype='uint8')
        depth_place = np.zeros((depth_h, depth_w, channels), dtype='uint8')
        ir_place[place_ir:place_ir + ir_h, :, :] = ir_frame #for no homography
        rgb_place = rgb_frame #for no homography
        depth_place[place_depth:place_depth + depth_h, :, :] = depth_frame #for no homog or RGB/D view

    # inference
    if (human_detection):
        strong_rect, medium_rects, rgb_rects, ir_rects, pos_depth, rgb_place = human_detector.human_detector(rgb_frame, full_depth, full_ir)
    if (interaction_detection):
        rgb_place = interaction_detector.interaction_detector(rgb_frame, full_depth, full_ir)

    fr = fr + 1
    if fr%4 == 0: #print avg frame rate over 4 frames
        frame_rate=str(round(4/(time.time()- start), 2))
        start = time.time()
    cv2.putText(depth_place, frame_rate, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # display and write video
    disp = np.hstack((depth_place, ir_place, rgb_place))
    cv2.imshow("live", disp)
    
    if (rec): #save frames
        f += 1
        print ("frame No.", f)
        rgb_vid.write(rgb_frame)
        ir_vid.write(ir_frame)
        depth_vid.write(depth_frame)
        np.save(ir_name+str(f),full_ir)
        np.save(depth_name+str(f),full_depth)
        np.save(rgb_name+str(f),rgb_frame)

    if k == 27:  # esc key
        done = True
    if k == 49:  # 1 key
        human_detection = False
        interaction_detection = False
        homography_setting = 1
    if k == 50:  # 2 key
        human_detection = False
        interaction_detection = False
        homography_setting = 2
    if k == 48:  # 0 key
        human_detection = False
        interaction_detection = False
        homography_setting = 0
    if k == 104: #h key
        homography_setting = 2
        human_detection = True
    if k == 105: #i key
        homography_setting = 2
        interaction_detection = True
    if (k == 114 and not rec): # r key
        makedir()
        f = 0
        print("recording No.", recording)
        rec = True
    if (k == 115 and rec): # s key
        rec = False
        rgb_vid.release()
        ir_vid.release()
        depth_vid.release()
        print("recording stopped and videos saved")

# release resources and destoy windows
rgb_stream.stop()
depth_stream.stop()
openni2.unload()
cv2.destroyWindow("live")
if (interaction_detection):
    interaction_detector.end_classification()
print("Average frame rate:", fr/(time.time()-full_start))
print ("Completed video generation using {} codec". format(fourcc))
