from configparser import ConfigParser, RawConfigParser
import os
import time
from turtle import update
from cvat import AutoDownloadOnCVAT, AutoUploadOnCVAT, CVATCookies
from dataset_process import CVATDatasetProcess, DataMerge
from models.cvat_models import EngineTask, create_session as cvat_create_session
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

    def download(self, status='OD_Initialized'):
        session = ai_create_session()
        while True:
            session.commit()
            task = session.query(IriRecord).filter(IriRecord.status == status).order_by(IriRecord.update_time.desc()).first()
            if task != None:
                break
        session.close()
        task_id = task.task_id
        project_name = task.project
        cvat = AutoDownloadOnCVAT(self.api_path, self.annotation_format)
        cvat.download(project_name, task_id, self.cookie.cookies)
        task_list = cvat.search_task_ids(task_id, project_name)
        for each in task_list:
            cvat.download(project_name, each.task_id, self.cookie.cookies)
        
        return task

    def cvat_datasets_process(self):
        origin_path = self.configs['MoveOptions']['OriginPath']
        dataset_path = self.configs['MoveOptions']['TargetPath']
        copy_dataset = CVATDatasetProcess(
            origin_dir_path=origin_path,
            dataset_dir_path=dataset_path
        )
        copy_dataset.auto_run()

        return copy_dataset.dataset_folder

    def dataset_merge(self, dataset_folder, comp_type):
        basicline_dir = self.configs['MoveOptions']['BasicLineDir']
        data_merge = DataMerge(dataset_folder, basicline_dir, comp_type)
        data_merge.add_basicline()

    def iri_record_status_update(self, iri_record_id, status, od_training_status=None):
        session = ai_create_session()
        session.commit()
        session.query(IriRecord).filter(
            IriRecord.id == iri_record_id
        ).update({
            "status": status,
            "od_training": od_training_status
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

    def run(self):
        # Init Task
        task = self.iri_record_check_status()

        # Upload
        self.parsers()
        self.get_cookies()
        self.iri_record_status_update(task.id, 'Upload imagewith log on going')
        self.upload(task.id)
        self.iri_record_status_update(task.id, 'Upload imagewith log finish')

        # Download
        task = self.download()

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


if __name__ == '__main__':
    main = Main()
    while True:
        main.run()
