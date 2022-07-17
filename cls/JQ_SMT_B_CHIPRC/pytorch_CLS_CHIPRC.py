import numpy as np
import cv2, datetime
import os, shutil
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as trns
from torch.utils.data import DataLoader
from PIL import Image



def cls_inference(model_cls, input_images, class_names=['NG','OK'], mean=[0.2682, 0.2322, 0.2276], std=[0.2368, 0.2380, 0.2188], OUTPUT_IMG = '1'):
    OK_length = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transforms = trns.Compose([
                trns.Resize((224, 224)), 
                trns.ToTensor(), 
                trns.Normalize(
                mean=mean, 
                std=std)])

    img = Image.open(input_images).convert("RGB")
    #img.show()
    image_tensor = transforms(img)
    image_tensor = image_tensor.unsqueeze(0)
    image = DataLoader(
        image_tensor,
        batch_size=1,
        shuffle=True
    )
    for inputs in image:
        inputs = inputs.to(device)
        output = model_cls(inputs)
        output = nn.functional.softmax(output, dim=1)
        p, preds = torch.max(output, 1)
        print('Predict:',class_names[preds[0]],', value:',float(p[0]))

            

    return class_names[preds[0]]        



def load_model_cls(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_cls = torch.load(cls_model_path, map_location=device)

    return model_cls


if __name__ == "__main__":
    from os.path import isdir
    from os import makedirs
    import glob

    cls_model_path = r'E:\USI\spyder\MobileNet\JQ_CHIPRC_ORG\model_zoo\cls_ORG_model_20220617.pt'

    model_cls = load_model_cls(cls_model_path)
    
    image_folder_path = r'E:\USI\spyder\MobileNet\JQ_CHIPRC_identify\result\JQ_20220620_2103-253901-30B\chiprc_single'
    save_image_folder_path = r'E:\USI\spyder\MobileNet\JQ_CHIPRC_ORG\results\JQ_20220620_2103-253901-30B\single'
    image_type = '*.jpg'
    check_image_path = image_folder_path + '/' + image_type
    image_available = glob.glob(check_image_path)
    for index in range(len(image_available)):
        print(image_available[index])
        print('now = ', index+1, '/', len(image_available))
        ###
        AI_result = cls_inference(model_cls, image_available[index])
        print('AI predict = ', AI_result)

        image_1 = cv2.imread(image_available[index])
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
        image_name_include_type = os.path.basename(image_available[index])
        image_name = os.path.splitext(image_name_include_type)[0]
        if AI_result == 'NG':
            save_path = save_image_folder_path + '\\' + 'NG\\'  
            if not isdir(save_path):
                makedirs(save_path)
            cv2.imwrite(save_path + image_name_include_type , cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR))
        elif AI_result == 'OK':
            save_path = save_image_folder_path + '\\' + 'OK\\'  
            if not isdir(save_path):
                makedirs(save_path)
            cv2.imwrite(save_path + image_name_include_type , cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR))        
        else:
            save_path = save_image_folder_path + '\\' + 'other\\'  
            if not isdir(save_path):
                makedirs(save_path)
            cv2.imwrite(save_path + image_name_include_type, cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR))
    print('Finish process')
    