import numpy as np
import os
import shutil

def start():
    global curdir
    print("Type directory containing frame folders or nothing to use current working directory.")
    curdir = raw_input()
    if (curdir == ""):
        curdir = os.getcwd()

start()
while (1):
    print("Type starting frame number")
    startframe = raw_input()
    print("Type ending frame number")
    endframe = raw_input()
    print("Type name of action (foldername)")
    foldername = raw_input()
    newdir = curdir+"/"+foldername
    oldmask = os.umask(000)
    os.mkdir(newdir, 0777)
    os.mkdir(newdir+"/rgb_full_vid", 0777)
    os.mkdir(newdir+"/depth_full_vid", 0777)
    os.mkdir(newdir+"/ir_full_vid", 0777)
    os.umask(oldmask)
    
    try: #load and display frame
        for i in range(int(startframe),int(endframe)+1):
            shutil.copy(curdir+"/rgb_full_vid/rgb_frame_"+str(i)+".npy",newdir+"/rgb_full_vid")
            shutil.copy(curdir+"/depth_full_vid/depth_frame_"+str(i)+".npy",newdir+"/depth_full_vid")
            shutil.copy(curdir+"/ir_full_vid/ir_frame_"+str(i)+".npy",newdir+"/ir_full_vid")
        print("Success.")

    except: #print out valid frames
        files = os.listdir(curdir+"/rgb_full_vid/")
        for i in range(len(files)):
            files[i] = files[i].split('_')[2].strip(".npy")
        files.sort(key=int)
        print("Error copying. Valid frames are:", files)

