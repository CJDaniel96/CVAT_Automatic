"""
Created on Wed Nov 24 14:30:17 2021

@author: tw029589
"""
import cv2
import imutils
import os, time
import os.path
import numpy as np
from xml.dom import minidom
from tqdm import tqdm 

class extact_labels_to_imgs:
    _defaults = {'resize_to' : None , #(32, 32)
                 'xml_file' : "yolo/pre_process/cfg/xml_file.txt",
                 'object_xml_file' : "yolo/pre_process/cfg/xml_object.txt"}

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"   

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)

    def check_folder_path(self, extract_to, path):
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)
            os.makedirs(path + '\\' + 'yolo_config')
            os.makedirs(path + '\\' + 'txt')


    def chkEnv(self, extract_to, imgFolder, xmlFolder):
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)
            print("no {} folder, created.".format(extract_to))

        if(not os.path.exists(imgFolder)):
            print("There is no such folder {}".format(imgFolder))
            quit()

        if(not os.path.exists(xmlFolder)):
            print("There is no such folder {}".format(xmlFolder))
            quit()

        if(not os.path.exists(self.xml_file)):
            print("There is no xml file in {}".format(self.xml_file))
            quit()

        if(not os.path.exists(self.object_xml_file)):
            print("There is no object xml file in {}".format(self.object_xml_file))
            quit()

    def getLabels(self, imgFile, xmlFile):
        labelXML = minidom.parse(xmlFile)
        labelName = []
        labelXmin = []
        labelYmin = []
        labelXmax = []
        labelYmax = []
        totalW = 0
        totalH = 0
        countLabels = 0

        tmpArrays = labelXML.getElementsByTagName("name")
        for elem in tmpArrays:
            labelName.append(str(elem.firstChild.data))

        tmpArrays = labelXML.getElementsByTagName("xmin")
        for elem in tmpArrays:
            # print(elem.firstChild.data)
            # labelYmax.append(int(elem.firstChild.data))
            labelXmin.append(int(float(elem.firstChild.data)))

        tmpArrays = labelXML.getElementsByTagName("ymin")
        for elem in tmpArrays:
            # print(elem.firstChild.data)
            # labelYmax.append(int(elem.firstChild.data))
            labelYmin.append(int(float(elem.firstChild.data)))

        tmpArrays = labelXML.getElementsByTagName("xmax")
        for elem in tmpArrays:
            # print(elem.firstChild.data)
            # labelYmax.append(int(elem.firstChild.data))
            labelXmax.append(int(float(elem.firstChild.data)))

        tmpArrays = labelXML.getElementsByTagName("ymax")
        for elem in tmpArrays:
            # print(elem.firstChild.data)
            # labelYmax.append(int(elem.firstChild.data))
            labelYmax.append(int(float(elem.firstChild.data)))

        return labelName, labelXmin, labelYmin, labelXmax, labelYmax

    def write_lale_images(self, label, img, extract_to, filename):
        writePath = os.path.join(extract_to, label)
        # print("WRITE:", writePath)

        if not os.path.exists(writePath):
            os.makedirs(writePath)

        if(self.resize_to is not None):
            img = cv2.resize(img, self.resize_to)

        cv2.imwrite(os.path.join(writePath, filename), img)

    def extact_labels_to_imgs(self, imgFolder, xmlFolder, extract_to):
        print('=============== cropped img ===================')
        self.chkEnv(extract_to, imgFolder, xmlFolder)
        i = 0
        for file in tqdm(os.listdir(imgFolder)):
            # try:
            if True :
                filename, file_extension = os.path.splitext(file)
                file_extension = file_extension.lower()
                if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".JPG"):
                    print("Processing: ", os.path.join(imgFolder, file))

                    if not os.path.exists(os.path.join(xmlFolder, filename+".xml")):
                        print("Cannot find the file {} for the image.".format(os.path.join(xmlFolder, filename+".xml")))

                    else:
                        image_path = os.path.join(imgFolder, file)
                        xml_path = os.path.join(xmlFolder, filename+".xml")
                        labelName, labelXmin, labelYmin, labelXmax, labelYmax = self.getLabels(image_path, xml_path)

                        orgImage = cv2.imread(image_path)
                        image = orgImage.copy()
                        # print(filename)
                        for id, label in enumerate(labelName):
                            # print(labelXmin)
                            # print(labelYmin)
                            # print(labelXmax)
                            # print(labelYmax)
                            cv2.rectangle(image, (labelXmin[id], labelYmin[id]), (labelXmax[id], labelYmax[id]), (0,255,0), 2)
                            label_area = orgImage[labelYmin[id]:labelYmax[id], labelXmin[id]:labelXmax[id]]
                            label_img_filename = filename + "_" + str(id) + ".jpg"
                            # print(label_img_filename)
                            self.write_lale_images(labelName[id], label_area, extract_to, label_img_filename)
            # except:
            #     print('no file exit'+file)
                    #cv2.imshow("Image", imutils.resize(image, width=700))
                    # k = cv2.waitKey(1)
