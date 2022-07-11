"""
Created on Wed Nov 24 14:30:17 2021

@author: tw029589
"""

#%%
import cv2
import numpy as np
from tqdm import tqdm 
import glob, os, datetime
from PIL import Image

path = r'E:\Tylor\raw_dataset\2021-12-27_PCIE_1020+1109\images\*.JPG'
path = glob.glob(path)

for img_name in tqdm(path):
    
    img = Image.open(img_name)
    img_name = img_name.replace('.JPG','.jpg')
    img.save(img_name,"jpg")






# %%
import cv2
import numpy as np
from tqdm import tqdm 
import glob, os, datetime
from PIL import Image
import shutil

path1 = glob.glob(r'E:\Tylor\raw_dataset\2021-12-27_PCIE_1020\*\*.JPG')
path1 = list(filter(lambda x: '.JPG' in x, path1))
path2 = glob.glob(r'E:\Tylor\raw_dataset\2021-12-27_PCIE_1020\*\*.jpg')
path2 = list(filter(lambda x: '.jpg' in x, path2))
path3 = glob.glob(r'E:\Tylor\raw_dataset\2021-12-27_PCIE_1109\*\*.JPG')
path3 = list(filter(lambda x: '.JPG' in x, path3))
path4 = glob.glob(r'E:\Tylor\raw_dataset\2021-12-27_PCIE_1109\*\*.jpg')
path4 = list(filter(lambda x: '.jpg' in x, path4))

path = path1 + path2 + path3 + path4

output_path = r'E:\Tylor\raw_dataset\2021-12-27_PCIE_1020+1109\images'

for img in tqdm(path) : 
    img_name = img.split('\\')[-1]
    intput = img
    output = shutil.copyfile(intput, output_path + '\\' + img_name)
    # print(img_name)


# for img in path : 
#     os.remove(img)
# %%
