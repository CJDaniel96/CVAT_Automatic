import argparse
import cv2, datetime , time
import os, shutil
import configparser
import torch
import torch.nn as nn
from xml.dom.minidom import Document
from pathlib import Path
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import (check_img_size, increment_path, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords)
from utils.plots import Annotator, colors, save_one_box
from torchvision import transforms as trns
from torch.utils.data import DataLoader
from PIL import Image
import get_foldername_file as parser
import os
from os.path import isdir
from os import makedirs
import glob


average_yolov5_time = 0


class OD_models:
    def __init__(self, cfg):
        self.label_cls = ['NG', 'OK']
        self.model_od_para = {}
        self.model_cls_para = {}
        self.model_cls_path = ''
        self.cfg = cfg

    def init_config(self):
        config = configparser.ConfigParser()
        config.read(self.cfg)
        # model_od
        global project_global , name_global
        project_global = config.get('model_od','project')
        name_global = config.get('model_od','name')
        #self.model_od_para['project'] = config.get('model_od','project') 
        #self.model_od_para['name'] = config.get('model_od','name')  
        self.model_od_para['weights'] = config['model_od']['weights']
        self.model_od_para['conf_thres'] = config.getfloat('model_od','conf_thres') 
        self.model_od_para['iou_thres'] = config.getfloat('model_od','iou_thres') 
        self.model_od_para['classes'] = None
        self.model_od_para['agnostic_nms'] = config.getboolean('model_od','agnostic_nms') 
        self.model_od_para['max_det'] = config.getint('model_od','max_det') 
        self.model_od_para['line_thickness'] = config.getint('model_od','line_thickness')     
        self.model_od_para['save_conf'] = config.getboolean('model_od','save_conf') 
        self.model_od_para['hide_labels'] = config.getboolean('model_od','hide_labels')
        self.model_od_para['hide_conf'] = config.getboolean('model_od','hide_conf')   
        self.model_cls_para['imgsz_cls'] = config.getint('model_cls','imgsz_cls')


    def model_predict(self, img_path, model_od, imgsz, save_dir , OUTPUT_IMG = '1' ):
        AI_result = 'NG'
        OD_result = 'nothing'
        #project = self.model_od_para['project']
        #print("type : " , type(project))
        #name = self.model_od_para['name']
        conf_thres = self.model_od_para['conf_thres']
        iou_thres = self.model_od_para['iou_thres']
        classes = self.model_od_para['classes']
        agnostic_nms = self.model_od_para['agnostic_nms']
        max_det = self.model_od_para['max_det']
        line_thickness = self.model_od_para['line_thickness']
        names = model_od.names
        save_conf = self.model_od_para['save_conf']
        
        defect_name_xml = []
        defect_position_xml = []

        #save_dir = Path(save_dir_str)
        start_make_dir_time = time.time()
        
        #(save_dir / 'labels' ).mkdir(parents=True, exist_ok=True)  # make dir
        
        #print(img_path)
        end_make_dir_time = time.time()
        print("make dir time : " , end_make_dir_time - start_make_dir_time)
        
        obj_info = []
        dataset = LoadImages(img_path, img_size=imgsz,\
                             stride=model_od.stride, auto=model_od.pt)
        for path, im, im0s, vid_cap, s in dataset:
            # load img & transform
            print('path : ', path)
            output_crop_COMP = []
            output_crop_GAP  = []
            

            im = torch.from_numpy(im).to('cpu')
            im = im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            # pred
            global average_yolov5_time
            start_yolov5_start = time.time()
            pred = model_od(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres\
                , iou_thres, classes, agnostic_nms, max_det=max_det)   # <===
           
            end_yolov5_start = time.time()
            total_tolov5_time = end_yolov5_start - start_yolov5_start
            print("The yolov5 time used to execute classification is given below :" , total_tolov5_time)
            average_yolov5_time = average_yolov5_time + total_tolov5_time
            
            for i, det in enumerate(pred):
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)
                    #save_path = str(save_dir / p.name)
                    #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]
                    #print(s)
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    imc = im0.copy()
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                        defect_position_xml = det[:, :4].tolist()
                        for *xyxy, conf, cls in reversed(det):
                            # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.model_od_para['save_conf'] else (cls, *xywh)  # label format
                            
                            # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.model_od_para['hide_labels'] else (names[c] if self.model_od_para['hide_conf'] else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            # Output Crop
                            #print(label)
                            xyxy = torch.tensor(xyxy).clone().detach().view(-1, 4)
                            b = xyxy2xywh(xyxy)  # boxes
                            b[:, 2:] = b[:, 2:] * 1.02 + 10  # box wh * gain + pad
                            xyxy = xywh2xyxy(b).long()
                            clip_coords(xyxy, imc.shape)
                            crop = imc[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1)]
                            
                            defect_name_xml.insert(0,label)
                            
                            if label == 'COMP':
                                output_crop_COMP.append(crop)
                            elif label == 'GAP':
                                output_crop_GAP.append(crop)
                            else:
                                OD_result = 'BBOXwithNGfeature'
                               
        #print(model_od.names)
        print("im0s" , im0s.shape)
        print("defect_name_xml : ",defect_name_xml)
        print("defect_position_xml : ",defect_position_xml)
        im_ori_shape = im0s.shape
        #output_xml(im_ori_shape, path, defect_name_xml, defect_position_xml, save_image_path)
        img_name = path.split('\\')[-1]
        if OUTPUT_IMG == '1':
            now       = datetime.datetime.now()
            Date      = now.strftime("%Y-%m-%d")
            TEMP_PATH = save_dir
            SAVE_PATH = TEMP_PATH + '/' + 'jpgxml' + '/' 
            if not os.path.exists(SAVE_PATH):                        
                os.makedirs(SAVE_PATH)
            img_name = os.path.basename(img_path)
            shutil.copy2(img_path, SAVE_PATH + img_name)
            self.output_xml(im_ori_shape, img_name, defect_name_xml, defect_position_xml, SAVE_PATH)
            
            imgname = os.path.basename(img_path).split(".")[0]
            image_1 = cv2.imread(img_path)
            
            for idx, location in enumerate(defect_position_xml):
                save_path = save_dir + '\\' + str(defect_name_xml[idx])
                if not isdir(save_path):
                    makedirs(save_path)            
                pad_box = location
                print('==>',pad_box)
                cropped_image = image_1[int(pad_box[1]):int(pad_box[3]),int(pad_box[0]):int(pad_box[2]),:]
                
                file_path = save_path + '\\' + imgname + '_' + str(idx) + '.jpg'
                cv2.imwrite(file_path, cropped_image)             
            
            
            
            #self.cropimage(img_path, defect_name_xml, defect_position_xml, SAVE_PATH)
        
        return AI_result
    
    
    def cropimage(img_path, defect_name, defect_position, save_image_folder_path):
        imgname = os.path.basename(img_path).split(".")[0]
        image_1 = cv2.imread(img_path)
        
        for idx, location in enumerate(defect_position):
            save_path = save_image_folder_path + '\\' + str(defect_name[idx])
            if not isdir(save_path):
                makedirs(save_path)            
            pad_box = location
            cropped_image = image_1[pad_box[0]:pad_box[2],pad_box[1]:pad_box[3],:]
            file_path = save_path + '\\' + imgname + '_' + str(idx) + '.jpg'
            cv2.imwrite(file_path, cropped_image) 
            
    
    def output_xml(self, image_size, image_name, defect_name, defect_position, save_image_path):
            # Create empty root
            doc = Document()
            # Create root node
            root = doc.createElement('annotation')
            doc.appendChild(root)
            # Create folder node
            folder = doc.createElement('folder')
            folder_text = doc.createTextNode(save_image_path)
            folder.appendChild(folder_text)
            root.appendChild(folder)
            # Create filename node
            filename = doc.createElement('filename')
            filename_text = doc.createTextNode(image_name)
            filename.appendChild(filename_text)
            root.appendChild(filename)
            # Create path node
            path = doc.createElement('path')
            path_text = doc.createTextNode(save_image_path + image_name)
            path.appendChild(path_text)
            root.appendChild(path)
            # Create image size node
            size = doc.createElement('size')
            width = doc.createElement('width')
            width_text = doc.createTextNode(str(image_size[1]))
            width.appendChild(width_text)
            size.appendChild(width)
            height = doc.createElement('height')
            height_text = doc.createTextNode(str(image_size[0]))
            height.appendChild(height_text)
            size.appendChild(height)
            depth = doc.createElement('depth')
            depth_text = doc.createTextNode(str(image_size[2]))
            depth.appendChild(depth_text)
            size.appendChild(depth)
            root.appendChild(size)
            # Create object node
            if defect_name != None or defect_position != None:
                #import pdb;pdb.set_trace()
                for name_list, box_list in zip(defect_name, defect_position):
                    xml_object = doc.createElement('object')
                    # defect name
                    name = doc.createElement('name')
                    name_text = doc.createTextNode(name_list)
                    name.appendChild(name_text)
                    xml_object.appendChild(name)
                    # bndbox
                    bndbox = doc.createElement('bndbox')
                    print("box_list" , box_list)
                    # xmin
                    xmin = doc.createElement('xmin')
                    xmin_text = doc.createTextNode(str(int(box_list[0])))
                    xmin.appendChild(xmin_text)
                    bndbox.appendChild(xmin)
                    # ymin
                    ymin = doc.createElement('ymin')
                    ymin_text = doc.createTextNode(str(int(box_list[1])))
                    ymin.appendChild(ymin_text)
                    bndbox.appendChild(ymin)
                    # xmax
                    xmax = doc.createElement('xmax')
                    xmax_text = doc.createTextNode(str(int(box_list[2])))
                    xmax.appendChild(xmax_text)
                    bndbox.appendChild(xmax)
                    # ymax
                    ymax = doc.createElement('ymax')
                    ymax_text = doc.createTextNode(str(int(box_list[3])))
                    ymax.appendChild(ymax_text)
                    bndbox.appendChild(ymax)
                    xml_object.appendChild(bndbox)
                    root.appendChild(xml_object)
            if '.jpeg' in image_name:
                xml_name = image_name[0:-5] + '.xml'
            else:
                xml_name = image_name[0:-4] + '.xml'
            with open(save_image_path + '\\' + xml_name, 'w') as xml:
                doc.writexml(xml, indent='\t', newl='\n', addindent='\t', encoding='utf-8')

def load_model_od(model_path):
    # model_od
    model_od = DetectMultiBackend(model_path,\
                                    device=select_device('cpu'),\
                                    dnn=False)
    imgsz = check_img_size(640, s=model_od.stride)

    return model_od, imgsz

def get_file(image_folder_path):
    images_folder = []
    for root, _, files in os.walk(image_folder_path):
        for each in files:
            if '.jpg' in each:
                images_folder.append(os.path.join(root, each))
    return images_folder

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--od-model-path')
    arg_parser.add_argument('--image-folder-path')
    arg_parser.add_argument('--save-image-folder-path')
    arg_parser.add_argument('--cfg-path')
    args = arg_parser.parse_args()

    od_model_path = args.od_model_path
    image_folder_path = args.image_folder_path
    save_image_folder_path = args.save_image_folder_path
    cfg_path = args.cfg_path

    model = OD_models(cfg_path)
    model.init_config()
    model_od, imgsz = load_model_od(od_model_path)

    index = 0
    do_image = get_file(image_folder_path)
    for path_name in do_image:
        
        print('now = ', index, '/', len(do_image))
        print('image path = ', path_name)
        AI_result = model.model_predict(path_name, model_od, imgsz, save_image_folder_path)   # <===
        print('AI predict = ', AI_result)
        print('')
        
        index = index +1
    print('Finish process')

if __name__ == "__main__":
    main()