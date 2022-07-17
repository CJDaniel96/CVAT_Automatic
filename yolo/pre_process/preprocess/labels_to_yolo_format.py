"""
Created on Wed Nov 24 14:30:17 2021

@author: tw029589
"""
import glob, os, time, cv2
import os.path
from shutil import copyfile
from xml.dom import minidom
from os.path import basename
from tqdm import tqdm 

class labels_to_yolo_format:
    _defaults = {'resize_to' : None , #(32, 32)
                 'xml_file' : "yolo/pre_process/cfg/xml_file.txt",
                 'object_xml_file' : "yolo/pre_process/cfg/xml_object.txt"}


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)

    def check_folder_path(self, saveYoloPath):
        if not os.path.exists(saveYoloPath):
            print("no {} folder, created.".format(saveYoloPath))
            os.makedirs(saveYoloPath)

    def transferYolo(self, xmlFilepath, imgFilepath, saveYoloPath, classList):
        global imgFolder

        img_file, img_file_extension = os.path.splitext(imgFilepath)
        img_filename = basename(img_file)
        # print(imgFilepath)

        if(xmlFilepath is not None):
            img = cv2.imread(imgFilepath)
            imgShape = img.shape
            # print (img.shape)
            img_h = imgShape[0]
            img_w = imgShape[1]

            labelXML = minidom.parse(xmlFilepath)
            labelName = []
            labelXmin = []
            labelYmin = []
            labelXmax = []
            labelYmax = []
            totalW = 0
            totalH = 0
            countLabels = 0

            tmpArrays = labelXML.getElementsByTagName("filename")
            for elem in tmpArrays:
                filenameImage = elem.firstChild.data

            tmpArrays = labelXML.getElementsByTagName("name")
            for elem in tmpArrays:
                labelName.append(str(elem.firstChild.data))

            tmpArrays = labelXML.getElementsByTagName("xmin")
            for elem in tmpArrays:
                labelXmin.append(int(float(elem.firstChild.data)))

            tmpArrays = labelXML.getElementsByTagName("ymin")
            for elem in tmpArrays:
                # labelYmin.append(int(elem.firstChild.data))
                labelYmin.append(int(float(elem.firstChild.data)))

            tmpArrays = labelXML.getElementsByTagName("xmax")
            for elem in tmpArrays:
                # labelXmax.append(int(elem.firstChild.data))
                labelXmax.append(int(float(elem.firstChild.data)))

            tmpArrays = labelXML.getElementsByTagName("ymax")
            for elem in tmpArrays:
                # labelYmax.append(int(elem.firstChild.data))
                labelYmax.append(int(float(elem.firstChild.data)))

            yoloFilename = os.path.join(saveYoloPath, img_filename + ".txt")
            # print("writeing to {}".format(yoloFilename))

            with open(yoloFilename, 'a') as the_file:
                i = 0
                for className in labelName:
                    if(className in classList):
                        # print(className)
                        classID = classList[className]
                        x = (labelXmin[i] + (labelXmax[i]-labelXmin[i])/2) * 1.0 / img_w 
                        y = (labelYmin[i] + (labelYmax[i]-labelYmin[i])/2) * 1.0 / img_h
                        w = (labelXmax[i]-labelXmin[i]) * 1.0 / img_w
                        h = (labelYmax[i]-labelYmin[i]) * 1.0 / img_h

                        the_file.write(str(classID) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
                        i += 1

        else:
            yoloFilename = os.path.join(saveYoloPath ,img_filename + ".txt")
            print("writeing negative file to {}".format(yoloFilename))

            with open(yoloFilename, 'a') as the_file:
                the_file.write('')

        the_file.close()


    def trans_to_yolo_format(self, imgFolder, xmlFolder, saveYoloPath, negFolder, classList):
        fileCount = 0
        print('trans dataset to yolo format')
        self.check_folder_path(saveYoloPath)
        for file in tqdm(os.listdir(imgFolder)):
            # print(file)
            filename, file_extension = os.path.splitext(file)
            file_extension = file_extension.lower()

            if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
                imgfile = os.path.join(imgFolder, file)
                xmlfile = os.path.join(xmlFolder ,filename + ".xml")

                if(os.path.isfile(xmlfile)):
                    
                    # print("id:{}".format(fileCount))
                    # print("processing {}".format(imgfile))
                    # print("processing {}".format(xmlfile))
                    fileCount += 1

                    self.transferYolo( xmlfile, imgfile, saveYoloPath, classList)
                    copyfile(imgfile, os.path.join(saveYoloPath ,file))

        if(os.path.exists(negFolder)):
            print('trans neg sample')
            for file in tqdm(os.listdir(negFolder)):
                filename, file_extension = os.path.splitext(file)
                file_extension = file_extension.lower()
                imgfile = os.path.join(negFolder ,file)

                if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
                    self.transferYolo( None, imgfile, "")
                    copyfile(imgfile, os.path.join(saveYoloPath ,file))
