from configparser import ConfigParser, RawConfigParser
from datetime import datetime
import os
import time
from cvat import AutoDownloadOnCVAT, AutoUploadOnCVAT, CVATCookies
from dataset_process import CLSDatasetProcess, CVATDatasetProcess, DataMerge, ImageAugmentor, WithODCLSDatasetProcess
from models.ai_models import IriRecord, create_session as ai_create_session
from cls.JQ_SMT_B_CHIPRC import train as JQ_SMT_B_CHIPRC_train
from yolo.pre_process.yolo_preprocess import YOLOPreProcess


class Main:
    def __init__(self) -> None:
        self.images_folder = None
        self.xml_zip = None
        self.api_path = None
        self.annotation_format = None
        self.cookie = None
        self.configs = ConfigParser()
        self.dataset_folder = None

    def parsers(self):
        self.configs.read('settings/config.ini')

        raw_configs = RawConfigParser()
        raw_configs.read('settings/config.ini')

        self.images_folder = self.configs['CVATOptions']['ImagesFolder']
        self.xml_path = self.configs['CVATOptions']['XMLPath']
        self.api_path = self.configs['CVATOptions']['APIPath']
        self.annotation_format = raw_configs['CVATOptions']['AnnotationFormat']

    def get_cookies(self):
        cvat_cookies = CVATCookies(self.api_path)
        self.cookie = cvat_cookies.get_login_cookies()

    def upload(self, iri_record):
        cvat = AutoUploadOnCVAT(iri_record, self.api_path, self.annotation_format)
        task_id, task_name = cvat.upload(self.images_folder, self.xml_path, self.cookie)
        self.configs.set('CVATOptions', 'TaskId', str(task_id))
        self.configs.set('CVATOptions', 'TaskName', task_name)
        with open('settings/config.ini', 'w') as f:
            self.configs.write(f)

        session = ai_create_session()
        session.commit()
        session.query(IriRecord).filter(
            IriRecord.id == iri_record.id
        ).update({
            "task_id": task_id,
            "task": task_name
        })
        session.commit()
        session.close()

    def download(self, iri_record):
        cvat = AutoDownloadOnCVAT(self.api_path, self.annotation_format)
        cvat.download(iri_record.task, iri_record.task_id, self.cookie.cookies)
        task_list = cvat.search_task_ids(iri_record.task_id, iri_record.group_type)
        for each in task_list:
            cvat.download(iri_record.task, each.task_id, self.cookie.cookies)

    def cvat_datasets_process(self, iri_record):
        origin_path = self.configs['ODMoveOptions']['OriginPath']
        dataset_path = self.configs['ODMoveOptions']['TargetPath']
        copy_dataset = CVATDatasetProcess(
            iri_record=iri_record,
            origin_dir_path=origin_path,
            dataset_dir_path=dataset_path
        )
        copy_dataset.auto_run()

        return copy_dataset.dataset_folder

    def cls_datasets_process(self, iri_record):
        origin_path = self.configs['CLSMoveOptions']['OriginPath']
        dataset_dir = self.configs['CLSMoveOptions']['TargetDir']
        dataset_path = dataset_dir + '\\' + iri_record.project + '\\datasets'
        copy_dataset = CLSDatasetProcess(
            iri_record=iri_record,
            dataset_dir_path=dataset_path,
            origin_dir_path=origin_path
        )
        copy_dataset.auto_run()

        return copy_dataset.dataset_folder_list, copy_dataset.ng_category_list, copy_dataset.ok_category_list

    def with_od_cls_datasets_process(self, iri_record, origin_path):
        dataset_dir = self.configs['CLSMoveOptions']['TargetDir']
        dataset_path = os.path.join(dataset_dir, iri_record.project, 'datasets')
        copy_dataset = WithODCLSDatasetProcess(
            iri_record=iri_record,
            dataset_dir_path=dataset_path, 
            origin_dir_path=origin_path
        )
        copy_dataset.run()

        return copy_dataset.dataset_folder_list

    def od_dataset_merge(self, dataset_folder, comp_type):
        basicline_dir = self.configs['Basicline']['BasicLineDir']
        data_merge = DataMerge(basicline_dir, comp_type)
        data_merge.od_add_basicline(dataset_folder)

    def cls_dataset_merge(self, dataset_folder_list, comp_type, ng_categories, ok_categories):
        basicline_dir = self.configs['Basicline']['BasicLineDir']
        data_merge = DataMerge(basicline_dir, comp_type)
        data_merge.cls_add_basicline(dataset_folder_list, ng_categories, ok_categories)

    def images_augmentor(self, dataset_folder_list):
        sub_folder_list = [
            'train\\OK',
            'train\\NG',
            'val\\OK',
            'val\\NG'
        ]
        aug = ImageAugmentor()
        for dataset_folder in dataset_folder_list:
            for each in sub_folder_list:
                aug.src_path = os.path.join(dataset_folder, each)
                aug.augmentor()
                aug.images_resave_and_rename()

    def iri_record_status_update(self, iri_record_id, status, od_training_status=None, cls_training_status=None):
        session = ai_create_session()
        session.commit()
        session.query(IriRecord).filter(
            IriRecord.id == iri_record_id
        ).update({
            "status": status,
            "od_training": od_training_status,
            "cls_training": cls_training_status
        })
        session.commit()
        session.close()

    def iri_record_check_status(self, status='Init', interval=60):
        print('Waiting for ' + status + ' task')
        session = ai_create_session()
        while True:
            task = session.query(IriRecord).filter(IriRecord.status == status).order_by(IriRecord.update_time).first()
            if task != None:
                break
            else:
                time.sleep(interval)
        session.close()

        return task

    def check_od_or_cls(self, interval=60):
        print('Check status is OD or CLS')
        session = ai_create_session()
        while True:
            iri_record = session.query(IriRecord).filter(IriRecord.status.in_(['OD_Initialized', 'CLS_Initialized'])).order_by(IriRecord.update_time.desc()).first()
            if iri_record != None:
                break
            else:
                time.sleep(interval)
        session.close()

        return iri_record

    def cls_training(self, project, cls_dataset_folder, model_save_folder):
        model_save_dir = self.configs['CLSMoveOptions']['TargetDir'] + '\\' + project + '\\' + model_save_folder
        if project == 'JQ_SMT_B_CHIPRC':
            JQ_SMT_B_CHIPRC_train.create_save_dir(model_save_dir)
            JQ_SMT_B_CHIPRC_train.train(cls_dataset_folder, model_save_dir)

    def run(self):
        # Init Task
        # iri_record = self.iri_record_check_status()

        # # Upload
        self.parsers()
        self.get_cookies()
        # self.iri_record_status_update(iri_record.id, 'Upload imagewith log on going')
        # self.upload(iri_record)
        # self.iri_record_status_update(iri_record.id, 'Upload imagewith log finish')

        # Decide OD_Initialized first or CLS_Initialized first
        iri_record = self.check_od_or_cls()
        if iri_record.status == 'OD_Initialized':
            # Download
            self.download(iri_record)

            # Unzip Data
            self.dataset_folder = self.cvat_datasets_process(iri_record)
            class_name = iri_record.project
            self.od_dataset_merge(self.dataset_folder, class_name)

            # Pre-process
            fullpath = os.getcwd() + self.dataset_folder[1:]
            work_path = os.getcwd()
            self.iri_record_status_update(iri_record.id, 'Trigger training for OD', 'Running')
            self.yolo_preprocess = YOLOPreProcess(iri_record, fullpath)
            self.yolo_preprocess.run()

            # Yolo Training
            os.chdir(work_path + '/yolo/training_code/yolov5')
            self.iri_record_status_update(iri_record.id, 'Training for OD', 'Running')
            os.system('python train.py --batch 8 --epochs 300 --data ./data/' + class_name + '.yaml' + ' --cfg ./models/' + class_name + '.yaml')
            os.chdir(work_path)

        # CLS
        if iri_record.status == 'CLS_Initialized':
            class_name = iri_record.project
            # With OD Training
            if iri_record.od_training == 'Done':
                self.iri_record_status_update(iri_record.id, 'Trigger training for CLS', 'Done', 'Running')
                cls_dataset_folder_list = self.with_od_cls_datasets_process(iri_record, self.dataset_folder)

            # Without OD Training
            else:
                self.iri_record_status_update(iri_record.id, 'Trigger training for CLS', '-', 'Running')
                self.download(iri_record)
                cls_dataset_folder_list, cls_ng_category_list, cls_ok_category_list = self.cls_datasets_process(iri_record)
                self.cls_dataset_merge(cls_dataset_folder_list, class_name, cls_ng_category_list, cls_ok_category_list)

            # Augmentor
            self.images_augmentor(cls_dataset_folder_list)

            # CLS Training
            self.iri_record_status_update(iri_record.id, 'Training for CLS', '-', 'Running')
            for cls_dataset_folder in cls_dataset_folder_list:
                model_save_folder = 'saved_models\\' + 'save_' + class_name + '_' + cls_dataset_folder.split('\\')[-1] + '_' + datetime.now().strftime('%Y%m%d') + '\\'
                self.cls_training(class_name, cls_dataset_folder, model_save_folder)


if __name__ == '__main__':
    main = Main()
    while True:
        main.run()
