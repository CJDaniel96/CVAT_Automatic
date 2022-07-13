from datetime import datetime
from configparser import ConfigParser
from sklearn.model_selection import train_test_split
from models.ai_models import CategoryMapping, create_session as ai_create_session

import os
import shutil
import zipfile
import argparse
import xml.etree.ElementTree as ET


class DatasetProcessing:
    """
    Attributes:
        get_dataset_folder:       get the dataset folder
        create_dataset_folder:    create the dataset folder of preserve the downloaded data.
        zip_dir_path:             the .zip file path of downlaoded data.
        move_data:                move data to the raw dataset folder.
        remove_unzip_file:        remove the unzip file from the download path.
        unzip:                    unzip the downloaded .zip file.
        run:                      the run main program entrance.
        auto_run:                 auto run parse the origin path to unzip and move the data.
    """
    def __init__(self, zip_path=None, dataset_dir_path=None, origin_dir_path=None) -> None:
        self.zip_path = zip_path
        self.dataset_dir_path = dataset_dir_path
        self.origin_dir_path = origin_dir_path

    def get_dataset_folder_name(self, dataset_classes, date_time) -> str:
        date = datetime.strptime(date_time, '%Y%m%d').strftime('%Y-%m-%d')
        dataset_folder = self.dataset_dir_path + '\\' + date + '_' + dataset_classes

        return dataset_folder

    def mkdir(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
            print('Create ' + path + ' Success!')

    def create_dataset_folder(self):
        dataset_classes = ''
        zip_name = self.zip_path.split('\\')[-1]
        for each in zip_name.split('_')[:-1]:
            dataset_classes += each + '_'
        dataset_classes = dataset_classes[:-1]
        date_time = zip_name.split('_')[-1].split('.')[0]
        dataset_folder = self.get_dataset_folder_name(dataset_classes, date_time)
        self.mkdir(dataset_folder)

        return dataset_folder

    def zip_dir_path(self):
        zip_dir = ''
        for each in self.zip_path.split('\\')[:-1]:
            zip_dir += each + '\\'
        return zip_dir

    def move_data(self, zip_dir, dataset_folder):
        images_dir_path = zip_dir + 'JPEGImages\\'
        xmls_dir_path = zip_dir + 'Annotations\\'
        raw_dataset_images_folder = dataset_folder + '\\images\\'
        raw_dataset_xmls_folder = dataset_folder + '\\xml\\'

        if os.path.isdir(images_dir_path):
            shutil.move(images_dir_path, raw_dataset_images_folder)
        if os.path.isdir(xmls_dir_path):
            shutil.move(xmls_dir_path, raw_dataset_xmls_folder)

        print('Copy finish!')

    def remove_unzip_file(self, zip_dir, zip_path):
        if os.path.isdir(zip_dir + '\\Annotations'):
            shutil.rmtree(zip_dir + '\\Annotations')
        if os.path.isdir(zip_dir + '\\ImageSets'):
            shutil.rmtree(zip_dir + '\\ImageSets')
        if os.path.isdir(zip_dir + '\\JPEGImages'):
            shutil.rmtree(zip_dir + '\\JPEGImages')
        if os.path.isfile(zip_dir + '\\labelmap.txt'):
            os.remove(zip_dir + '\\labelmap.txt')
        if os.path.isfile(zip_path):
            os.remove(zip_path)

        print('Clean Unzip Files Finish!')

    def unzip(self):
        if zipfile.is_zipfile(self.zip_path):
            zip_file = zipfile.ZipFile(self.zip_path, 'r')
            zip_file.extractall()
            print('Unzip ' + self.zip_path.split('\\')[-1] + ' Finish!')
            return True
        else:
            print('The zip file is not exist!')
            return False

    def run(self):
        if self.unzip():
            zip_dir = self.zip_dir_path()
            dataset_folder = self.create_dataset_folder()
            self.move_data(zip_dir, dataset_folder)
            self.remove_unzip_file(zip_dir, self.zip_path)

    def auto_run(self):
        for each in os.listdir(self.origin_dir_path):
            if each[-4:] == '.zip':
                self.zip_path = os.path.join(self.origin_dir_path, each)
                self.run()

class CVATDatasetProcess(DatasetProcessing):
    """Unzip and move the downloaded CVAT data to directive path.

    Attributes:
        create_dataset_folder:    create the dataset folder of preserve the downloaded data.
        zip_dir_path:             the .zip file path of downlaoded data.
        move_data:                move data to the raw dataset folder.
        remove_unzip_file:        remove the unzip file from the download path.
        unzip:                    unzip the downloaded .zip file.
        run:                      the run main program entrance.
        auto_run:                 auto run parse the origin path to unzip and move the data.
    """
    def __init__(self, zip_path=None, dataset_dir_path=None, origin_dir_path=None) -> None:
        super(CVATDatasetProcess, self).__init__(zip_path, dataset_dir_path, origin_dir_path)

    def get_dataset_folder_name(self, dataset_classes, date_time) -> str:
        date = datetime.strptime(date_time, '%Y%m%d').strftime('%Y-%m-%d')
        if 'JQ' in dataset_classes:
            dataset_folder = self.dataset_dir_path + '\\' + 'JQCHIPRC' + \
                '\\' + date + '_' + dataset_classes
        elif 'CHIP' in dataset_classes:
            dataset_folder = self.dataset_dir_path + '\\' + 'CHIPRC' + \
                '\\' + date + '_' + dataset_classes
        elif 'PCIE' in dataset_classes:
            dataset_folder = self.dataset_dir_path + '\\' + 'PCIE' + \
                '\\' + date + '_' + dataset_classes
        elif 'SOLDER' in dataset_classes:
            dataset_folder = self.dataset_dir_path + '\\' + 'SOLDER' + \
                '\\' + date + '_' + dataset_classes

        return dataset_folder


class DataMerge:
    def __init__(self, dataset_folder, basicline_dir, comp_type):
        self.dataset_folder = dataset_folder
        self.basicline_dir = basicline_dir
        self.comp_type = comp_type

    def add_basicline(self):
        basicline = ''
        basicline_list = sorted(
            [
                each.split('_')
                for each in os.listdir(self.basicline_dir + '\\' + self.comp_type)
                if 'basicline' in each
            ],
            key=lambda x: x[1],
            reverse=True
        )
        for each in basicline_list[0]:
            basicline += each + '_'
        basicline = basicline[:-1]
        images_dir = self.basicline_dir + '\\' + \
            self.comp_type + '\\' + basicline + '\\images'
        xml_dir = self.basicline_dir + '\\' + self.comp_type + '\\' + basicline + '\\xml'
        for each in os.listdir(images_dir):
            shutil.copyfile(
                images_dir + '\\' + each,
                self.dataset_folder + '\\images\\' + each
            )
        for each in os.listdir(xml_dir):
            shutil.copyfile(
                xml_dir + '\\' + each,
                self.dataset_folder + '\\xml\\' + each
            )

        print('Add BasicLine Finish!')


class CLSDatasetProcess(DatasetProcessing):
    def __init__(self, site, lines, group_type, project, zip_path=None, dataset_dir_path=None, origin_dir_path=None) -> None:
        super().__init__(zip_path, dataset_dir_path, origin_dir_path)
        self.site = site
        self.lines = lines
        self.group_type = group_type
        self.project = project

    @property
    def line(self):
        return eval(self.lines)

    def get_categories(self):
        session = ai_create_session()
        session.commit()
        category_mapping = session.query(CategoryMapping).filter(
            CategoryMapping.site == self.site, 
            CategoryMapping.line.in_(self.line), 
            CategoryMapping.group_type == self.group_type, 
            CategoryMapping.project == self.project
        ).all()
        session.close()

        return [eval(category.ng_category) for category in category_mapping], [eval(category.ok_category) for category in category_mapping], [category.line for category in category_mapping]

    def categories_processing(self, ng_categories, ok_categories, category_lines):
        category_list = []
        line_name_list = []
        for ng_category, ok_category, category_line in zip(ng_categories, ok_categories, category_lines):
            if repr(ng_category) + ', ' + repr(ok_category) not in category_list:
                category_list.append(repr(ng_category) + ', ' + repr(ok_category))
                line_name_list.append(category_line)
            else:
                line_name_list[category_list.index(repr(ng_category) + ', ' + repr(ok_category))] = line_name_list[category_list.index(repr(ng_category) + ', ' + repr(ok_category))] + category_line

        
        return category_list, line_name_list

    def category_split(self, category):
        ng_category = eval(category)[0]
        ok_category = eval(category)[1]

        return ng_category, ok_category

    def parse_xml_ng_ok(self, xml, ng_category, ok_category) -> bool:
        type_list = []
        tree = ET.parse(xml)
        root = tree.getroot()
        for obj in root.findall('object'):
            type_list.append(obj.find('name').text)
        if list(set(type_list) & set(ok_category)) == []:
            return False
        elif list(set(type_list) & set(ng_category)) != []:
            return False
        else:
            return True

    def create_sub_dataset_folder(self, dataset_folder, line_name):
        sub_dataset_folder = os.path.join(dataset_folder, line_name)
        self.mkdir(sub_dataset_folder)

        return sub_dataset_folder

    def create_ng_ok_folder(self, dataset_folder):
        self.mkdir(dataset_folder + '\\train')
        self.mkdir(dataset_folder + '\\val')
        self.mkdir(dataset_folder + '\\train\\NG')
        self.mkdir(dataset_folder + '\\train\\OK')
        self.mkdir(dataset_folder + '\\val\\NG')
        self.mkdir(dataset_folder + '\\val\\OK')

    def move(self, src_folder, dst_folder):
        for each in src_folder:
            shutil.move(each, dst_folder)

    def move_data(self, zip_dir, dataset_folder, ng_category, ok_category):
        ng_folder = []
        ok_folder = []
        for xml in os.listdir(zip_dir + '\\Annotations'):
            xml_path = zip_dir + 'Annotations\\' + xml
            img = xml[:-4] + '.jpg'
            img_path = zip_dir + 'JPEGImages\\' + img
            if self.parse_xml_ng_ok(xml_path, ng_category, ok_category):
                ok_folder.append(img_path)
            else:
                ng_folder.append(img_path)
        if ok_folder:
            ok_train, ok_val = train_test_split(ok_folder)
            self.move(ok_train, dataset_folder + '\\train\\OK')
            self.move(ok_val, dataset_folder + '\\val\\OK')
        if ng_folder:
            ng_train, ng_val = train_test_split(ng_folder)
            self.move(ng_train, dataset_folder + '\\train\\NG')
            self.move(ng_val, dataset_folder + '\\val\\NG')

    def run(self):
        if self.unzip():
            zip_dir = self.zip_dir_path()
            ng_categories, ok_categories, category_lines = self.get_categories()
            category_list, line_name_list = self.categories_processing(ng_categories, ok_categories, category_lines)
            for each_category, each_line_name in zip(category_list, line_name_list):
                dataset_folder = self.create_dataset_folder()
                dataset_folder = self.create_sub_dataset_folder(dataset_folder, each_line_name)
                self.create_ng_ok_folder(dataset_folder)
                ng_category, ok_category = self.category_split(each_category)
                self.move_data(zip_dir, dataset_folder, ng_category, ok_category)
                self.remove_unzip_file(zip_dir, self.zip_path)

def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto-run', action='store_true',
                        help='Auto to unzip files to the target path that YOLO can catch.')
    parser.add_argument('--zip-path', help='The CVAT .zip file path.')
    parser.add_argument(
        '--dataset-path', help='The path of YOLO catch the dataset.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argsparser()
    if args.auto_run:
        configs = ConfigParser()
        configs.read('settings/config.ini')
        origin_path = configs['MoveOptions']['OriginPath']
        dataset_dir_path = configs['MoveOptions']['TargetPath']
        copy_dataset = CVATDatasetProcess(
            origin_dir_path=origin_path,
            dataset_dir_path=dataset_dir_path
        )
        copy_dataset.auto_run()
        configs.set(
            'PreProcessOptions',
            'rawdatapath',
            copy_dataset.dataset_folder
        )
        with open('settings/config.ini', 'w') as f:
            configs.write(f)
    else:
        zip_path = args.zip_path
        dataset_path = args.dataset_path
        copy_dataset = CVATDatasetProcess(
            zip_path=zip_path,
            dataset_dir_path=dataset_path
        )
        copy_dataset.run()
