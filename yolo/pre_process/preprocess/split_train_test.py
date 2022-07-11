"""
Created on Wed Nov 24 14:30:17 2021

@author: tw029589
"""
import random
import glob, os, datetime
import os.path
from shutil import copyfile

class split_train_test:
    _defaults = {'fileList' : [],
                 'fileList_colab' : []}
                #  'outputTrainFile' : os.path.join(cfgFolder,"train.txt"),
                #  'outputTestFile' : os.path.join(cfgFolder ,"test.txt"),
                #  'output_colabTrainFile' : os.path.join(cfgFolder,"train_colab.txt"),
                #  'output_colabTestFile' : os.path.join(cfgFolder ,"test_colab.txt")}

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)

    def split_train_test_txt(self, cfgFolder, saveYoloPath, testRatio, outputTrainFile, outputTestFile):
        if not os.path.exists(cfgFolder):
            os.makedirs(cfgFolder)

        for file in os.listdir(saveYoloPath):
            filename, file_extension = os.path.splitext(file)
            file_extension = file_extension.lower()

            if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
                self.fileList.append(os.path.join(saveYoloPath, file))
                # fileList_colab.append(colab_yolo_path+file)

        print("total image files: ", len(self.fileList))

        testCount = int(len(self.fileList) * testRatio)
        trainCount = len(self.fileList) - testCount

        a = range(len(self.fileList))
        test_data = random.sample(a, testCount)
        train_data = [x for x in a if x not in test_data]

        print ("Train:{} images".format(len(train_data)))
        print("Test:{} images".format(len(test_data)))

        with open(outputTrainFile, 'w') as the_file:
            for i in train_data:
                the_file.write(self.fileList[i] + "\n")
        the_file.close()

        with open(outputTestFile, 'w') as the_file:
            for i in test_data:
                the_file.write(self.fileList[i] + "\n")
        the_file.close()

        # with open(output_colabTrainFile, 'w') as the_file:
        #     for i in train_data:
        #         the_file.write(fileList_colab[i] + "\n")
        # the_file.close()

        # with open(output_colabTestFile, 'w') as the_file:
        #     for i in test_data:
        #         the_file.write(fileList_colab[i] + "\n")
        # the_file.close()


    def fit_yolo_training_format(self, path, output_path):
        file_name = path.split('\\')[-1].split('_')[-1] 
        folder_name = datetime.datetime.today().strftime("%Y-%m-%d_" + file_name)
        output_path =output_path + '\\' + folder_name
        if not os.path.exists(output_path):
            os.makedirs(output_path + '\\images\\train')
            os.makedirs(output_path + '\\images\\val')
            os.makedirs(output_path + '\\labels\\train')
            os.makedirs(output_path + '\\labels\\val')

        train_txt = path + '\\yolo_config\\train.txt'
        txt = train_txt
        txt = list(open(txt, mode='r'))
        txt = list(map(lambda x: x.replace('\n',''),txt))
        for img_file in txt:
            img_name = img_file.split('\\')[-1]
            img_outputpath = output_path + '\\images\\train\\' + img_name
            copyfile(img_file, img_outputpath)
            
            label_file = img_file.replace(img_file[-4:], '.txt')
            txt_name = label_file.split('\\')[-1]
            txt_outputpath = output_path + '\\labels\\train\\' + txt_name
            copyfile(label_file, txt_outputpath)
        
        test_txt = path + '\\yolo_config\\test.txt'
        txt = test_txt
        txt = list(open(txt, mode='r'))
        txt = list(map(lambda x: x.replace('\n',''),txt))
        for img_file in txt:
            img_name = img_file.split('\\')[-1]
            img_outputpath = output_path + '\\images\\val\\' + img_name
            copyfile(img_file, img_outputpath)
            # print(img_file)
            label_file = img_file.replace(img_file[-4:], '.txt')
            txt_name = label_file.split('\\')[-1]
            txt_outputpath = output_path + '\\labels\\val\\' + txt_name
            copyfile(label_file, txt_outputpath)
        return output_path


    def split_train_test(self, saveYoloPath, testRatio, path, output_path):
        cfgFolder = path + '\\' + 'yolo_config'
        outputTrainFile = os.path.join(cfgFolder,"train.txt")
        outputTestFile = os.path.join(cfgFolder ,"test.txt")
        self.split_train_test_txt(cfgFolder, saveYoloPath, testRatio, outputTrainFile, outputTestFile)
        output_path = self.fit_yolo_training_format(path, output_path)
        return output_path