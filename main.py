from configparser import ConfigParser, RawConfigParser
from datetime import datetime
import os
import time
from cvat import AutoDownloadOnCVAT, AutoUploadOnCVAT, CVATCookies
from dataset_process import CLSDatasetProcess, CVATDatasetProcess, DataMerge, WithODCLSDatasetProcess
from models.ai_models import IriRecord, create_session as ai_create_session


class Main:
    def __init__(self) -> None:
        self.images_folder = None
        self.xml_zip = None
        self.project_name = None
        self.api_path = None
        self.annotation_format = None
        self.cookie = None
        self.configs = ConfigParser()

    def parsers(self):
        self.configs.read('settings/config.ini')

        raw_configs = RawConfigParser()
        raw_configs.read('settings/config.ini')

        self.images_folder = self.configs['CVATOptions']['ImagesFolder']
        self.xml_path = self.configs['CVATOptions']['XMLPath']
        self.project_name = self.configs['CVATOptions']['ProjectName']
        self.api_path = self.configs['CVATOptions']['APIPath']
        self.annotation_format = raw_configs['CVATOptions']['AnnotationFormat']

    def get_cookies(self):
        cvat_cookies = CVATCookies(self.api_path)
        self.cookie = cvat_cookies.get_login_cookies()

    def upload(self, iri_record_id):
        cvat = AutoUploadOnCVAT(self.api_path, self.annotation_format)
        task_id = cvat.upload(self.images_folder, self.xml_path, self.project_name, self.cookie)
        self.configs.set('CVATOptions', 'TaskId', str(task_id))
        with open('settings/config.ini', 'w') as f:
            self.configs.write(f)

        session = ai_create_session()
        session.query(IriRecord).filter(
            IriRecord.id == iri_record_id
        ).update({
            "task_id": task_id
        })
        session.close()

    def download(self, task):
        task_id = task.task_id
        project_name = task.project
        cvat = AutoDownloadOnCVAT(self.api_path, self.annotation_format)
        cvat.download(project_name, task_id, self.cookie.cookies)
        task_list = cvat.search_task_ids(task_id, project_name)
        for each in task_list:
            cvat.download(project_name, each.task_id, self.cookie.cookies)

    def cvat_datasets_process(self):
        origin_path = self.configs['ODMoveOptions']['OriginPath']
        dataset_path = self.configs['ODMoveOptions']['TargetPath']
        copy_dataset = CVATDatasetProcess(
            origin_dir_path=origin_path,
            dataset_dir_path=dataset_path
        )
        copy_dataset.auto_run()

        return copy_dataset.dataset_folder

    def cls_datasets_process(self, site, lines, group_type, project):
        origin_path = self.configs['CLSMoveOptions']['OriginPath']
        dataset_path = self.configs['CLSMoveOptions']['TargetPath']
        copy_dataset = CLSDatasetProcess(
            site=site,
            lines=lines,
            group_type=group_type,
            project=project,
            dataset_dir_path=dataset_path,
            origin_dir_path=origin_path
        )
        copy_dataset.auto_run()

        return copy_dataset.dataset_folder

    def without_od_cls_datasets_process(self, class_name, dataset_folder):
        origin_path = self.configs['CLSMoveOptions']['OriginPath']
        dataset_path = self.configs['CLSMoveOptions']['TargetPath']
        src = src + '/' + class_name + '/' + dataset_folder.split('\\')[-1]
        copy_dataset = WithODCLSDatasetProcess(dataset_path, origin_path)
        copy_dataset.run()

        return copy_dataset.dataset_folder

    def dataset_merge(self, dataset_folder, comp_type):
        basicline_dir = self.configs['Basicline']['BasicLineDir']
        data_merge = DataMerge(dataset_folder, basicline_dir, comp_type)
        data_merge.add_basicline()

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
            task = session.query(IriRecord).filter(IriRecord.status.in_(['OD_Initialized', 'CLS_Initialized'])).order_by(IriRecord.update_time.desc()).first()
            if task != None:
                break
            else:
                time.sleep(interval)
        session.close()

        return task

    def run(self):
        # Init Task
        task = self.iri_record_check_status()

        # Upload
        self.parsers()
        self.get_cookies()
        self.iri_record_status_update(task.id, 'Upload imagewith log on going')
        self.upload(task.id)
        self.iri_record_status_update(task.id, 'Upload imagewith log finish')

        # Decide OD_Initialized first or CLS_Initialized first
        task = self.check_od_or_cls()
        if task.status == 'OD_Initialized':
            # Download
            self.download(task)

            # Unzip Data
            dataset_folder = self.cvat_datasets_process()
            class_name = 'PCIE' if 'PCIE' in dataset_folder else 'SOLDER' if 'SOLDER' in dataset_folder else 'JQCHIPRC' if 'JQ' in dataset_folder else 'CHIPRC'
            self.dataset_merge(dataset_folder, class_name)

            # Pre-process
            fullpath = os.getcwd() + dataset_folder[1:]
            work_path = os.getcwd()
            self.iri_record_status_update(task.id, 'Trigger training for OD', 'Running')
            os.chdir(work_path + '/yolo/pre_process')
            os.system('python yolo_preprocess.py --path ' + fullpath)

            # Yolo Training
            os.chdir(work_path + '/yolo/training_code/yolov5')
            self.iri_record_status_update(task.id, 'Training for OD', 'Running')
            os.system('python train.py --batch 8 --epochs 300 --data ./data/' + class_name + '.yaml' + ' --cfg ./models/' + class_name + '.yaml')
            os.chdir(work_path)

        # CLS
        if task.status == 'CLS_Initialized':
            self.iri_record_status_update(task.id, 'Trigger training for CLS', '-', 'Running')
            # With OD Training
            if task.od_training == 'Done':
                dataset_folder = self.without_od_cls_datasets_process(class_name, dataset_folder)

            # Without OD Training
            else:
                self.download(task)
                dataset_folder = self.cls_datasets_process(task.site, task.line, task.group_type, task.project)

            # CLS Training
            os.chdir(work_path + '/cls/' + class_name)
            self.iri_record_status_update(task.id, 'Training for CLS', '-', 'Running')
            model_save_folder = 'save_' + class_name + '_ORG_' + datetime.now().strftime('%Y%m%d') + '/'
            os.system('python train.py --data-dir ' + dataset_folder + ' --model-save-dir ./saved_models/' + model_save_folder)


if __name__ == '__main__':
    main = Main()
    while True:
        main.run()
