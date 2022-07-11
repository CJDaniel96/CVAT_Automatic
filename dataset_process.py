import argparse
from configparser import ConfigParser
from datetime import datetime
import os
import shutil
import zipfile


class CVATDatasetProcess:
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

    def __init__(self, zip_path=None, dataset_dir_path=None, origin_dir_path=None):
        self.zip_path = zip_path
        self.dataset_dir_path = dataset_dir_path
        self.origin_dir_path = origin_dir_path
        self._dataset_folder = None

    def create_dataset_folder(self):
        dataset_classes = ''
        zip_name = self.zip_path.split('\\')[-1]
        for each in zip_name.split('_')[:-1]:
            dataset_classes += each + '_'
        dataset_classes = dataset_classes[:-1]
        zip_date = zip_name.split('_')[-1].split('.')[0]
        date_time = datetime.strptime(zip_date, '%Y%m%d').strftime('%Y-%m-%d')
        if 'JQ' in dataset_classes:
            dataset_folder = self.dataset_dir_path + '\\' + 'JQCHIPRC' + \
                '\\' + date_time + '_' + dataset_classes
        elif 'CHIP' in dataset_classes:
            dataset_folder = self.dataset_dir_path + '\\' + 'CHIPRC' + \
                '\\' + date_time + '_' + dataset_classes
        elif 'PCIE' in dataset_classes:
            dataset_folder = self.dataset_dir_path + '\\' + 'PCIE' + \
                '\\' + date_time + '_' + dataset_classes
        elif 'SOLDER' in dataset_classes:
            dataset_folder = self.dataset_dir_path + '\\' + 'SOLDER' + \
                '\\' + date_time + '_' + dataset_classes
        if not os.path.isdir(dataset_folder):
            os.mkdir(dataset_folder)

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
        if os.path.isdir(zip_dir + '\\ImageSets'):
            shutil.rmtree(zip_dir + '\\ImageSets')
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

    @property
    def dataset_folder(self):
        return self._dataset_folder

    def run(self):
        if self.unzip():
            zip_dir = self.zip_dir_path()
            self._dataset_folder = self.create_dataset_folder()
            self.move_data(zip_dir, self._dataset_folder)
            self.remove_unzip_file(zip_dir, self.zip_path)

    def auto_run(self):
        for each in os.listdir(self.origin_dir_path):
            if each[-4:] == '.zip':
                self.zip_path = os.path.join(self.origin_dir_path, each)
                self.run()


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
