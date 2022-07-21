import argparse
import os

from yolo.pre_process.preprocess.extract_all_label_to_imgs import extact_labels_to_imgs
from yolo.pre_process.preprocess.labels_to_yolo_format import labels_to_yolo_format
from yolo.pre_process.preprocess.split_train_test import split_train_test
from models.ai_models import CategoryMapping, create_session as ai_create_session


class YOLOPreProcess:
    def __init__(self, iri_record, raw_data_path, test_ratio=0.2):
        self.project = iri_record.project
        self.site = iri_record.site
        self.group_type = iri_record.group_type
        self.raw_data_path = raw_data_path
        self.test_ratio = test_ratio
        self.output_path = './yolo/training_code/Datasets/training_data/' + iri_record.project

    def get_class_list(self):
        session = ai_create_session()
        category_mapping = session.query(CategoryMapping.labels).filter(
            CategoryMapping.site == self.site,
            CategoryMapping.group_type == self.group_type,
            CategoryMapping.project == self.project
        ).first()

        return eval(category_mapping.labels)

    def run(self):
        extact_labels = extact_labels_to_imgs()
        trans_to_yolo = labels_to_yolo_format()
        split_traintest = split_train_test()
        class_list = self.get_class_list()

        img_folder = self.raw_data_path + '\\' + 'images'
        xml_folder = self.raw_data_path + '\\' + 'xml'
        extract_to = self.raw_data_path + '\\' + 'extract'
        save_yolo_path = self.raw_data_path + '\\' + 'txt'
        neg_folder = ''

        extact_labels.extact_labels_to_imgs(img_folder, xml_folder, extract_to)
        trans_to_yolo.trans_to_yolo_format(
            img_folder, xml_folder, save_yolo_path, neg_folder, class_list
        )
        folder = split_traintest.split_train_test(
            save_yolo_path, self.test_ratio, self.raw_data_path, self.output_path
        )
        print('End')

        folder = folder.split('\\')[-1]
        train_path = self.output_path + '/' + folder + '/images/train/'
        test_path = self.output_path + '/' + folder + '/images/val/'
        print('train:' + train_path)
        print('test:' + test_path)
        print('number of classes:' + str(len(class_list.keys())))

        data_yaml_path = './yolo/training_code/yolov5/data/' + self.project + '.yaml'
        with open(data_yaml_path, 'w') as f:
            f.writelines('names: ' + str(list(class_list.keys())) + '\n')
            f.writelines('nc: ' + str(len(class_list.keys())) + '\n')
            f.writelines(
                'train: ' + os.getcwd() +
                    '/yolo/training_code/Datasets/training_data/' +
                    self.project + '/' + folder + '/images/train/' + '\n'
            )
            f.writelines(
                'val: ' + os.getcwd() +
                    '/yolo/training_code/Datasets/training_data/' +
                    self.project + '/' + folder + '/images/val/' + '\n'
            )
