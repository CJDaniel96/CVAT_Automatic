from configparser import ConfigParser, RawConfigParser
from datetime import datetime
import os
import zipfile
import requests
import argparse
from models.ai_models import OdTrainingInfo, create_session as ai_create_session
from settings.command_line_instruction import instruction
from models.cvat_models import EngineProject, create_session as cvat_create_session


class CVATCookies:
    """Provides the Cookies from CVAT Platform    

    Attributes:
        get_login_cookies: A string cookies while login
    """

    def __init__(self,
                 api_path='http://10.0.13.80:8080/api/v1/',
                 login_json={
                     'username': 'admin',
                     'password': 'usi',
                 }):
        """inits methods varibles"""
        self.api_path = api_path
        self.login_json = login_json

    def get_login_cookies(self):
        """A string cookies while login.

        Return:
            Response: a string of key.
        """

        login_url = self.api_path + 'auth/login'
        login_cookies = requests.post(login_url, json=self.login_json)

        return login_cookies


class AutoUploadOnCVAT:
    """This is a programme to auto-upload ai labeled data to CVAT Platform

    Attributes:
        auth_header:                 Http headers
        task_create_info:            A dictonary of create task information.
        create_task:                 Create task on the CVAT
        task_data:                   the Images of upload the task.
        upload_task_data:            Upload the images to task on the CVAT.
        upload_tasks_annotations:    Upload the Images' annotations to task on the CVAT.
    """

    def __init__(self, iri_record, api_path='http://10.0.13.80:8080/api/v1/', annotation_format='PASCAL%20VOC%201.1'):
        """inits methods varibles"""
        self.iri_record = iri_record
        self.api_path = api_path
        self.annotation_format = annotation_format
        self._task_info = None
        self._auth_header = None
        self._task_data = {}
        self._task_name = None

    @property
    def auth_header(self):
        """Get http header.

        Return:
            Json: a dictonary of http header.
        """

        return self._auth_header

    @auth_header.setter
    def auth_header(self, token):
        """Set http header.

        Args:
            token (str): a login cookies key.
        """

        self._auth_header = {"Authorization": "Token " + token}

    @property
    def task_name(self):
        """Set and Get task Name

        Returns:
            str: the task name create on the CVAT.
        """
        line_name = ''
        serial_number = datetime.now().strftime('%Y%m%d%H%M%S')
        for each in eval(self.iri_record.line):
            line_name += each
        self._task_name = self.iri_record.group_type + '_' + line_name + '_' + serial_number

        return self._task_name

    def task_create_info(self):
        """make a dictonary of create task information.

        Args:
            project_id (int):    the project id of your want to create the task.
        Return: 
            Json:                a dictonary of create task information.
        Example:
            {
                "name": self._task_name,
                "labels": [],
                "project_id": project_id,
            }
        """

        return {
            "name": self.task_name,
            "labels": [],
            "project_id": self.iri_record.project_id
        }

    def create_task(self, auth_header, task_create_info):
        """Create task on the CVAT

        Args:
            auth_header (json):         the http header
            task_create_info (json):    create task information.
        Return:
            int:                        the task id of generation when create the task.
        """

        response = requests.post(
            self.api_path + "tasks", headers=auth_header, data=task_create_info)
        task_id = response.json()['id']
        print('Create Task ' + str(task_id) + ' Finish!')

        return task_id

    @property
    def task_data(self):
        """Get the Images of upload the task.

        Return:
            _task_data: a dictionary of upload Images informations.
        """

        return self._task_data

    @task_data.setter
    def task_data(self, folder):
        """Set the Images of upload the task.

        Args:
            folder (str): the images folder path.
        """

        for i, each in enumerate(os.listdir(folder)):
            image_file = (each, open(os.path.join(folder, each), "rb"))
            file_key = "client_files[{}]".format(i)
            self._task_data[file_key] = image_file

    def upload_task_data(self, task_id, auth_header, task_data):
        """Upload the images to task on the CVAT.

        Args:
            task_id (int):         the task id of create the task.
            auth_header (json):    Http headers.
            task_data (dict):      the dictionary of images.
        """

        upload_info = {"image_quality": 70}
        upload_url = self.api_path + 'tasks/' + str(task_id) + '/data'
        requests.post(upload_url, data=upload_info,
                      headers=auth_header, files=task_data)

        print('Upload Images to task ' + str(task_id) + ' Finish!')

    def zip_dir(self, path):
        zf = zipfile.ZipFile('{}.zip'.format(path), 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(path):
            for file_name in files:
                zf.write(os.path.join(root, file_name))

    def upload_tasks_annotations(self, task_id, auth_header, xml_path):
        """Upload the Images' annotations to task on the CVAT.

        Args:
            task_id (int):               the task id of create the task.
            auth_header (json):          Http headers.
            xml_path (str):              the annotation XML path.
        """
        zip_path = xml_path + '.zip'
        self.zip_dir(xml_path)
        uploaded_data = {
            "annotation_file": open(zip_path, "rb")
        }
        annotation_url = self.api_path + \
            'tasks/' + str(task_id) + '/annotations?format=' + \
            self.annotation_format
        requests.put(annotation_url, headers=auth_header, files=uploaded_data)

        print('Upload annotations to task ' + str(task_id) + ' Finish!')

    def upload(self, images_folder, xml_path, cookie):
        """Run uploading the images and xmls to CVAt

        Args:
            images_folder (str): The images folder path
            xml_path (str): The xml file path
            cookie (Response): THe login auths cookie
        """
        self.auth_header = cookie.json()['key']
        task_create_info = self.task_create_info()
        task_id = self.create_task(self.auth_header, task_create_info)
        self.task_data = images_folder
        self.upload_task_data(task_id, self.auth_header, self.task_data)
        self.upload_tasks_annotations(task_id, self.auth_header, xml_path)
        print('Upload Finish!')

        return task_id, self.task_name


class AutoDownloadOnCVAT:
    def __init__(self, api_path='http://10.0.13.80:8080/api/v1/', annotation_format='PASCAL%20VOC%201.1'):
        """inits methods varibles"""
        self.api_path = api_path
        self.annotation_format = annotation_format
        self.tasks_url = 'http://10.0.13.80:8080/api/v1/tasks'

    def get_latest_task(self, cookies):
        tasks_data = requests.get(self.tasks_url, cookies=cookies)
        task = tasks_data.json()['results'][0]

        return task

    def search_task_ids(self, task_id, comp_type, val_status='approve'):
        session = ai_create_session()
        od_training_info = session.query(OdTrainingInfo).filter(
            OdTrainingInfo.task_id < task_id,
            OdTrainingInfo.comp_type == comp_type,
            OdTrainingInfo.val_status == val_status
        ).all()
        session.close()

        return [task for task in od_training_info]

    def download(self, task_name, task_id, cookies):
        """Download the tasks images and annotations.

        Args:
            task_id (int): the task id on the CVAT.
            cookies (cookies): the login user cookies.
        """

        url = self.api_path + \
            "tasks/{}/dataset?format={}&action=download".format(
                task_id, self.annotation_format
            )
        while True:
            resp = requests.get(url, cookies=cookies)
            if len(resp.content) != 0:
                break
        zip_name = task_name + '.zip'
        with open(zip_name, 'wb') as zip_file:
            zip_file.write(resp.content)

        print('Download the task ' + str(task_id) + ' Finish!')


class Settings:
    """This is the auto upload and download settings.

    Keyword Arguments:
        - project_list (dict): The CVAT projects in the dictionary.

    Attributes:
        check_project_name:    Check project name exist in CVAT projects
        get_project_id:        Get project id
        argparser_settings:    The argparser settting arguments.
    """

    @classmethod
    def check_project_name(cls, project_name):
        """Check project name exist in CVAT projects

        Args:
            project_name (str): the project name

        Returns:
            boolean: project name is or not exist
        """

        session = cvat_create_session()
        projects = session.query(EngineProject).all()

        for project in projects:
            if project_name == project.name:
                return True

        return False

    @classmethod
    def get_project_id(cls, project_name):
        """Get project id

        Args:
            project_name (str): the project name

        Returns:
            int: project id
        """

        session = cvat_create_session()
        project_id = session.query(EngineProject.id).filter(
            EngineProject.name == project_name).first().id
        return project_id

    @classmethod
    def argparser_settings(cls):
        """The argparser settting arguments.

        Returns:
            args: the argparser arguments
        """

        parser = argparse.ArgumentParser()

        # arg help
        parser.add_argument(
            '--args',
            action='store_true',
            help='show all args and directions.'
        )

        # status
        parser.add_argument(
            '--upload',
            action='store_true',
            help='set true that your want to do upload action.'
        )
        parser.add_argument(
            '--download',
            action='store_true',
            help='set true that your want to do download action.'
        )

        # upload state command line script
        parser.add_argument(
            '--images-folder',
            default='',
            help='Type your want to upload images folder path.'
        )
        parser.add_argument(
            '--xml-zip',
            default='',
            help='Type your want to upload annotation xml zip path.'
        )
        parser.add_argument(
            '--task-name',
            default='',
            help='Type the task name your want to create.'
        )
        parser.add_argument(
            '--project-name',
            default='',
            help='Type the project name of your wanna download the task in which project'
        )
        # download state command line script
        # --project-name is the same as waht in upload state
        parser.add_argument(
            '--task-id',
            default='',
            help='Type the task id your want to download.'
        )

        # no must need to provide the script
        parser.add_argument(
            '--api-path',
            default='http://10.0.13.80:8080/api/v1/',
            help='Type the apu path to upload or download'
        )
        parser.add_argument(
            '--annotation-format',
            default='PASCAL%20VOC%201.1',
            help='Type the annotation format.'
        )

        parser.add_argument(
            '--run',
            action='store_true',
            help='auto run upload and download.'
        )

        args = parser.parse_args()

        return args


def manual_mode(args):
    """The Command Line Mode.

    Args:
        args: get the command line script.
              The command we have 10 types as following:

              0.  --args                  show all args and directions.

              1.  --upload                upload status (boolean).
              2.  --download              download status (boolean).

              upload state command line script(must provides if your choose upload):
              3.  --images-folder         upload images folder path.
              4.  --xml-zip               upload annotations XML zip path.
              5.  --task-name             upload the task name what your wanna create.
              6.  --project-name          upload the task belong to which project.

              download state command line script(must provides if your choose download):
              6.  --project-name          download the task belong to which project.
              7.  --task-id               download the task what id is.

              no must need to provide the script:
              8.  --api-path              provide the url
                                          default is 'http://10.0.13.80:8080/api/v1/'
              9.  --annotation-format     the task annotation format, 
                                          default is 'PASCAL%20VOC%201.1' (PASCAL VOC 1.1)
                                          because http parser blank is '%20'
    """

    cvat_cookies = CVATCookies(args.api_path)
    cookie = cvat_cookies.get_login_cookies()

    if args.upload == True:
        cvat = AutoUploadOnCVAT(args.api_path, args.annotation_format)
        task_id, task_name = cvat.upload(
            args.images_folder,
            args.xml_zip,
            args.task_name,
            args.project_name,
            cookie
        )
    elif args.download == True:
        cvat = AutoDownloadOnCVAT(args.api_path, args.annotation_format)
        cvat.download(args.project_name, args.task_id, cookie.cookies)


def auto_mode(action):
    """The auto mode to uplaod or download CVAT Images and xml files

    Args:
        mode (_type_): _description_
    """
    configs = ConfigParser()
    configs.read('settings/config.ini')

    raw_configs = RawConfigParser()
    raw_configs.read('settings/config.ini')

    images_folder = configs['CVATOptions']['ImagesFolder']
    xml_zip = configs['CVATOptions']['XMLZip']
    project_name = configs['CVATOptions']['ProjectName']
    api_path = configs['CVATOptions']['APIPath']
    annotation_format = raw_configs['CVATOptions']['AnnotationFormat']

    cvat_cookies = CVATCookies(api_path)
    cookie = cvat_cookies.get_login_cookies()

    if action == 'upload':
        cvat = AutoUploadOnCVAT(api_path, annotation_format)
        task_id, task_name = cvat.upload(images_folder, xml_zip, project_name, cookie)
        configs.set('CVATOptions', 'TaskId', task_id)
        with open('settings/config.ini', 'w') as f:
            configs.write(f)
    elif action == 'download':
        cvat = AutoDownloadOnCVAT(api_path, annotation_format)
        task_id = configs['CVATOptions']['TaskId']
        cvat.download(project_name, task_id, cookie.cookies)


def main(args):
    # show args instruction
    if args.args == True:
        instruction()
        return

    if args.upload is True:
        if not (args.images_folder and args.xml_zip and args.task_name and args.project_name):
            auto_mode(action='upload')
        else:
            manual_mode(args)
    if args.download is True:
        if not (args.project_name and args.task_id):
            auto_mode(action='download')
        else:
            manual_mode(args)


if __name__ == '__main__':
    args = Settings.argparser_settings()
    main(args)
