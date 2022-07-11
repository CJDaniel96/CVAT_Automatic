from configparser import ConfigParser
from datetime import datetime
import shutil
import yaml


class DataYaml:
    def __init__(self, yaml_file_path, label_name):
        self.yaml_file_path = yaml_file_path
        self.label_name = label_name

    def write(self, train_path, val_path, nc, names):
        with open(self.yaml_file_path, 'w') as f:
            yaml.dump({
                'train': train_path,
                'val': val_path,
                'nc': nc,
                'names': names,
            }, f)
        print('Finish!')

    def update(self, train_path, val_path):
        with open(self.yaml_file_path) as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        configs['train'] = train_path
        configs['val'] = val_path
        yaml.dump(configs, open(self.yaml_file_path, 'w'))

    def duplicate(self):
        dst_path = ''
        file_name = datetime.now().date().strftime('%Y-%m-%d') + \
            '_' + self.label_name + '.yaml'
        for each in self.yaml_file_path.split('\\')[:-1]:
            dst_path += each + '\\'
        dst_path += file_name
        shutil.copy2(self.yaml_file_path, dst_path)
