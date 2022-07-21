from datetime import datetime
from configparser import ConfigParser
import Augmentor
from sklearn.model_selection import train_test_split
from models.ai_models import CategoryMapping, create_session as ai_create_session
import os
import shutil
import zipfile
import argparse
import xml.etree.ElementTree as ET
from yolo.pre_process.preprocess.extract_all_label_to_imgs import extact_labels_to_imgs


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
    def __init__(self, iri_record, zip_path=None, dataset_dir_path=None, origin_dir_path=None) -> None:
        self.task_name = iri_record.task
        self.task_id = iri_record.task_id
        self.zip_path = zip_path
        self.dataset_dir_path = dataset_dir_path
        self.origin_dir_path = origin_dir_path
        self._dataset_folder = None

    @property
    def dataset_folder(self):
        return self._dataset_folder

    @dataset_folder.setter
    def dataset_folder(self, path):
        self._dataset_folder = path

    def mkdir(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
            print('Create ' + path + ' Success!')

    def create_dataset_folder(self):
        dataset_folder = os.path.join(self.dataset_dir_path, self.task_name)
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
            self.dataset_folder = self.create_dataset_folder()
            self.move_data(zip_dir, self.dataset_folder)
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
    def __init__(self, iri_record, zip_path=None, dataset_dir_path=None, origin_dir_path=None) -> None:
        super(CVATDatasetProcess, self).__init__(iri_record, zip_path, dataset_dir_path, origin_dir_path)
        self.project = iri_record.project

    def create_dataset_folder(self):
        dataset_folder = self.dataset_dir_path + '\\' + self.project + '\\' + self.task_name
        self.mkdir(dataset_folder)

        return dataset_folder

class DataMerge:
    def __init__(self, basicline_dir, comp_type):
        self.basicline_dir = basicline_dir
        self.comp_type = comp_type

    def get_basicline(self):
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

        return basicline

    def od_add_basicline(self, dataset_folder):
        basicline = self.get_basicline()
        images_dir = self.basicline_dir + '\\' + \
            self.comp_type + '\\' + basicline + '\\images'
        xml_dir = self.basicline_dir + '\\' + self.comp_type + '\\' + basicline + '\\xml'
        for each in os.listdir(images_dir):
            shutil.copyfile(
                images_dir + '\\' + each,
                dataset_folder + '\\images\\' + each
            )
        for each in os.listdir(xml_dir):
            shutil.copyfile(
                xml_dir + '\\' + each,
                dataset_folder + '\\xml\\' + each
            )

        print('Add BasicLine Finish!')

    def copyfile(self, src_file_list, dst_folder):
        for each in src_file_list:
            shutil.copyfile(each, dst_folder + '\\' + each.split('\\')[-1])

    def cls_add_basicline(self, cls_dataset_folder_list, ng_categories, ok_categories):
        basicline = self.get_basicline()
        images_dir = self.basicline_dir + '\\' + \
            self.comp_type + '\\' + basicline + '\\images'
        xml_dir = self.basicline_dir + '\\' + self.comp_type + '\\' + basicline + '\\xml'

        ok_folder = []
        ng_folder = []
        for cls_dataset_folder, ng_category, ok_category in zip(cls_dataset_folder_list, ng_categories, ok_categories):
            for each in os.listdir(xml_dir):
                img_name = each[:-4] + '.jpg'
                if CLSDatasetProcess.parse_xml_ng_ok(xml_dir + '\\' + each, ng_category, ok_category):
                    ok_folder.append(images_dir + '\\' + img_name)
                else:
                    ng_folder.append(images_dir + '\\' + img_name)
            if ok_folder:
                ok_train, ok_val = train_test_split(ok_folder)
                self.copyfile(ok_train, cls_dataset_folder + '\\train\\OK')
                self.copyfile(ok_val, cls_dataset_folder + '\\val\\OK')
            if ng_folder:
                ng_train, ng_val = train_test_split(ng_folder)
                self.copyfile(ng_train, cls_dataset_folder + '\\train\\NG')
                self.copyfile(ng_val, cls_dataset_folder + '\\val\\NG')


class CLSDatasetProcess(DatasetProcessing):
    def __init__(self, iri_record, zip_path=None, dataset_dir_path=None, origin_dir_path=None) -> None:
        super().__init__(iri_record, zip_path, dataset_dir_path, origin_dir_path)
        self.site = iri_record.site
        self.lines = iri_record.line
        self.group_type = iri_record.group_type
        self.project = iri_record.project
        self.img_folder_name = 'JPEGImages'
        self.xml_folder_name = 'Annotations'
        self._ng_category_list = []
        self._ok_category_list = []
        self._dataset_folder_list = []

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

        ng_categories = []
        ok_categories = []
        category_lines_list = []
        cls_models = []
        cls_mapping_categories = []
        for category in category_mapping:
            if eval(category.ng_category) not in ng_categories:
                ng_categories.append(eval(category.ng_category))
            if eval(category.ok_category) not in ok_categories:
                ok_categories.append(eval(category.ok_category))
                category_lines_list.append(category.line)
            else:
                category_lines_list[ok_categories.index(eval(category.ok_category))] += category.line
            if eval(category.cls_model) not in cls_models:
                cls_models.append(eval(category.cls_model))
            if eval(category.cls_mapping_category) not in cls_mapping_categories:
                cls_mapping_categories.append(eval(category.cls_mapping_category))

        return ng_categories, ok_categories, category_lines_list, cls_models, cls_mapping_categories

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

    @property
    def ng_category_list(self):
        return self._ng_category_list

    @ng_category_list.setter
    def ng_category_list(self, ng_list):
        self._ng_category_list = ng_list

    @property
    def ok_category_list(self):
        return self._ok_category_list

    @ok_category_list.setter
    def ok_category_list(self, ok_list):
        self._ok_category_list = ok_list

    @property
    def dataset_folder_list(self):
        return self._dataset_folder_list

    @dataset_folder_list.setter
    def dataset_folder_list(self, dataset_folder):
        self._dataset_folder_list.append(dataset_folder)

    @classmethod
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

    def create_cls_model_dataset_folder(self, dataset_folder, cls_model_name):
        cls_model_dataset_folder = os.path.join(dataset_folder, cls_model_name)
        self.mkdir(cls_model_dataset_folder)

        return cls_model_dataset_folder

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

    def default_split_ng_ok_imgs(self, src_dir, ng_category, ok_category):
        ng_folder = []
        ok_folder = []
        for xml in os.listdir(src_dir + '\\' + self.xml_folder_name):
            xml_path = os.path.join(src_dir, self.xml_folder_name, xml)
            img = xml[:-4] + '.jpg'
            img_path = os.path.join(src_dir, self.img_folder_name, img)
            if self.parse_xml_ng_ok(xml_path, ng_category, ok_category):
                ok_folder.append(img_path)
            else:
                ng_folder.append(img_path)

        return ng_folder, ok_folder

    def crop_split_ng_ok_imgs(self, src_dir, ng_category, ok_category, each_cls_mapping_category):
        ng_list = []
        ok_list = []
        ng_folder = []
        ok_folder = []
        img_folder = src_dir + '\\' + self.img_folder_name
        xml_folder = src_dir + '\\' + self.xml_folder_name
        extract_to = src_dir + '\\extract'

        extact_labels = extact_labels_to_imgs()
        extact_labels.extact_labels_to_imgs(img_folder, xml_folder, extract_to)

        for xml in os.listdir(xml_folder):
            xml_path = os.path.join(src_dir, self.xml_folder_name, xml)
            img_name = xml[:-4]
            if self.parse_xml_ng_ok(xml_path, ng_category, ok_category):
                ok_list.append(img_name)
            else:
                ng_list.append(img_name)

        for img in os.listdir(os.path.join(extract_to, each_cls_mapping_category)):
            img_path = os.path.join(extract_to, each_cls_mapping_category, img)
            for ng_img_name in ng_list:
                if ng_img_name in img:
                    ng_folder.append(img_path)
            for ok_img_name in ok_list:
                if ok_img_name in img:
                    ok_folder.append(img_path)

        return ng_folder, ok_folder

    def move_data(self, dataset_folder, ng_folder, ok_folder):
        if ok_folder:
            ok_train, ok_val = train_test_split(ok_folder)
            self.move(ok_train, dataset_folder + '\\train\\OK')
            self.move(ok_val, dataset_folder + '\\val\\OK')
        if ng_folder:
            ng_train, ng_val = train_test_split(ng_folder)
            self.move(ng_train, dataset_folder + '\\train\\NG')
            self.move(ng_val, dataset_folder + '\\val\\NG')

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

        if os.path.isdir(zip_dir + '\\extract'):
            shutil.rmtree(zip_dir + '\\extract')

        print('Clean Unzip Files Finish!')

    def run(self):
        if self.unzip():
            zip_dir = self.zip_dir_path()
            self.dataset_folder = self.create_dataset_folder()
            ng_categories, ok_categories, category_lines_list, cls_models, cls_mapping_categories = self.get_categories()
            self.ng_category_list, self.ok_category_list = ng_categories, ok_categories
            for ng_category, ok_category, category_line, cls_model_name, cls_mapping_category in zip(
                ng_categories, 
                ok_categories, 
                category_lines_list, 
                cls_models, 
                cls_mapping_categories
            ):
                self.dataset_folder = self.create_sub_dataset_folder(self.dataset_folder, category_line)
                for each_cls_model_name, each_cls_mapping_category in zip(cls_model_name, cls_mapping_category):
                    cls_model_dataset_folder = self.create_cls_model_dataset_folder(self.dataset_folder, each_cls_model_name)
                    self.create_ng_ok_folder(cls_model_dataset_folder)
                    self.dataset_folder_list = cls_model_dataset_folder
                    if each_cls_model_name == 'ORG':
                        ng_folder, ok_folder = self.default_split_ng_ok_imgs(zip_dir, ng_category, ok_category)
                        self.move_data(cls_model_dataset_folder, ng_folder, ok_folder)
                    else:
                        ng_folder, ok_folder = self.crop_split_ng_ok_imgs(zip_dir, ng_category, ok_category, each_cls_mapping_category)
                        self.move_data(cls_model_dataset_folder, ng_folder, ok_folder)
            self.remove_unzip_file(zip_dir, self.zip_path)


class WithODCLSDatasetProcess(CLSDatasetProcess):
    def __init__(self, iri_record, zip_path=None, dataset_dir_path=None, origin_dir_path=None):
        super().__init__(iri_record, zip_path, dataset_dir_path, origin_dir_path)
        self.xml_folder_name = 'xml'
        self.img_folder_name = 'images'

    def move(self, src_folder, dst_folder):
        for each in src_folder:
            dst_file = os.path.join(dst_folder, each.split('\\')[-1])
            shutil.copyfile(each, dst_file)

    def crop_split_ng_ok_imgs(self, ng_category, ok_category, each_cls_mapping_category):
        ng_list = []
        ok_list = []
        ng_folder = []
        ok_folder = []

        for xml in os.listdir(os.path.join(self.origin_dir_path, self.xml_folder_name)):
            xml_path = os.path.join(self.origin_dir_path, self.xml_folder_name, xml)
            img = xml[:-4]
            if self.parse_xml_ng_ok(xml_path, ng_category, ok_category):
                ok_list.append(img)
            else:
                ng_list.append(img)
        
        for img in os.listdir(os.path.join(self.origin_dir_path, 'extract', each_cls_mapping_category)):
            img_path = os.path.join(self.origin_dir_path, 'extract', each_cls_mapping_category, img)
            for ng_img_name in ng_list:
                if ng_img_name in img:
                    ng_folder.append(img_path)
            for ok_img_name in ok_list:
                if ok_img_name in img:
                    ok_folder.append(img_path)

    def run(self):
        self.dataset_folder = self.create_dataset_folder()
        ng_categories, ok_categories, category_lines_list, cls_models, cls_mapping_categories = self.get_categories()
        self.ng_category_list, self.ok_category_list = ng_categories, ok_categories
        for ng_category, ok_category, category_line, cls_model_name, cls_mapping_category in zip(
            ng_categories, 
            ok_categories, 
            category_lines_list, 
            cls_models, 
            cls_mapping_categories
        ):
            self.dataset_folder = self.create_sub_dataset_folder(self.dataset_folder, category_line)
            for each_cls_model_name, each_cls_mapping_category in zip(cls_model_name, cls_mapping_category):
                cls_model_dataset_folder = self.create_cls_model_dataset_folder(self.dataset_folder, each_cls_model_name)
                self.create_ng_ok_folder(cls_model_dataset_folder)
                self.dataset_folder_list = cls_model_dataset_folder
                if each_cls_model_name == 'ORG':
                    ng_folder, ok_folder = self.default_split_ng_ok_imgs(self.origin_dir_path, ng_category, ok_category)
                    self.move_data(cls_model_dataset_folder, ng_folder, ok_folder)
                else:
                    ng_folder, ok_folder = self.crop_split_ng_ok_imgs(self.origin_dir_path, ng_category, ok_category, each_cls_mapping_category)
                    self.move_data(cls_model_dataset_folder, ng_folder, ok_folder)


class ImageAugmentor:
    def __init__(self):
        self._src_path = None
        self.multi_thread = 5
        self._output_path = None
    
    @property
    def src_path(self):
        return self._src_path

    @src_path.setter
    def src_path(self, path):
        self._src_path = path

    @property
    def output_path(self):
        return os.path.join(self.src_path, 'output')

    def augmentor(self):
        length = len(os.listdir(self.src_path))
        p = Augmentor.Pipeline(self.src_path)
        p.random_brightness(probability = 0.5, min_factor = 0.8, max_factor = 1.2)
        p.random_contrast(probability = 0.5, min_factor = 0.8, max_factor = 1.2)
        p.random_color(probability = 0.5, min_factor = 0.8, max_factor = 1.2)
        p.flip_left_right(probability=0.5)
        p.flip_random(probability=0.5)
        p.flip_top_bottom(probability=0.5)
        p.rotate90(probability=0.5)
        p.rotate180(probability=0.5)
        p.rotate270(probability=0.5)
        p.sample(self.multi_thread * length)

    def images_resave_and_rename(self):
        count = 0
        for each in os.listdir(self.output_path):
            count += 1
            new_name = each.split('.')[0] + '_aug_' + str(count) + '.jpg'
            src_file = os.path.join(self.output_path, each)
            dst_file = os.path.join(self.src_path, new_name)
            shutil.move(src_file, dst_file)
        shutil.rmtree(self.output_path)


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
