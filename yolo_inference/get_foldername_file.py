# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:37:50 2020

@author: USER
"""


import os
import glob
from shutil import copyfile



def get_file(mypath):
    
    recording_folder = []
    recording_image = []
    
    for root, dirs, files in os.walk(mypath, topdown=False):
        for name in dirs:
             find_file_path = os.path.join(root, name)
             list_files = os.listdir(find_file_path)
             for filename in list_files:
                 recording_folder.append(os.path.basename(os.path.normpath(find_file_path)))
                 sourfile = os.path.join(find_file_path, filename)
                 if os.path.splitext(filename)[-1] == '.jpg' or os.path.splitext(filename)[-1] == '.png':
                     recording_image.append(sourfile)
    return recording_image, recording_folder



def get_top_side(recording_image):
    
    pair_light = []                     
    side = [s for s in recording_image if "side" in s]
    for side_path in side:
        name_without_light = os.path.basename(side_path)[0:-9]  # for AOI rawdata format
        name_without_light = os.path.basename(side_path)[0:-19] # for OP rapair rawdata format
        findsamename = [s for s in recording_image if name_without_light in s]
        if len(findsamename) > 2:
            if findsamename[0].find('side') and findsamename[1].find('top'):
                pair_light.append(findsamename[0:2])
            elif findsamename[0].find('top') and findsamename[1].find('side'):
                pair_light.append([findsamename[1],findsamename[0]])            
            elif findsamename[0].find('top') and findsamename[2].find('side'):
                pair_light.append([findsamename[2],findsamename[0]])
            elif findsamename[0].find('side') and findsamename[2].find('top'):
                pair_light.append([findsamename[0],findsamename[2]])
        elif len(findsamename) == 2:
            if findsamename[0].find('side') and findsamename[1].find('top'):
                pair_light.append(findsamename[0:2])
            elif findsamename[0].find('top') and findsamename[1].find('side'):
                pair_light.append([findsamename[1],findsamename[0]]) 

    return pair_light

if __name__ == "__main__":

    mypath = r'E:\USI\types\Holly\SZ\LS\02\temp'
    mypath = r'E:\USI\download\OP_Backup_classify_0930_tom\images\LS'
    
    
    recording_folder = []
    recording_image = []
    
    for root, dirs, files in os.walk(mypath, topdown=False):
        for name in dirs:
             find_file_path = os.path.join(root, name)
             list_files = os.listdir(find_file_path)
             for filename in list_files:
                 recording_folder.append(os.path.basename(os.path.normpath(find_file_path)))
                 sourfile = os.path.join(find_file_path, filename)
                 if os.path.splitext(filename)[-1] == '.jpg':
                     recording_image.append(sourfile)
                     
                     
    pair_light = []                     
    side = [s for s in recording_image if "side" in s]
    for side_path in side:
        name_without_light = os.path.basename(side_path)[0:-9]
        name_without_light = os.path.basename(side_path)[0:-19]
        findsamename = [s for s in recording_image if name_without_light in s]
        if len(findsamename) > 2:
            if findsamename[0].find('side') and findsamename[1].find('top'):
                pair_light.append(findsamename[0:2])
            elif findsamename[0].find('top') and findsamename[1].find('side'):
                pair_light.append([findsamename[1],findsamename[0]])            
            elif findsamename[0].find('top') and findsamename[2].find('side'):
                pair_light.append([findsamename[2],findsamename[0]])
            elif findsamename[0].find('side') and findsamename[2].find('top'):
                pair_light.append([findsamename[0],findsamename[2]])
        elif len(findsamename) == 2:
            if findsamename[0].find('side') and findsamename[1].find('top'):
                pair_light.append(findsamename[0:2])
            elif findsamename[0].find('top') and findsamename[1].find('side'):
                pair_light.append([findsamename[1],findsamename[0]])
                
                     
                     