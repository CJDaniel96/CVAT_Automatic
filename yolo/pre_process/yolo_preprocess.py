import argparse
import os

from preprocess.extract_all_label_to_imgs import extact_labels_to_imgs
from preprocess.labels_to_yolo_format import labels_to_yolo_format
from preprocess.split_train_test import split_train_test


def yolo_preprocess(path, testRatio=0.2):
    extact_labels = extact_labels_to_imgs()
    trans_to_yolo = labels_to_yolo_format()
    split_traintest = split_train_test()

    item = 'PCIE' if 'PCIE' in path else 'SOLDER' if 'SOLDER' in path else 'JQCHIPRC' if 'JQ' in path else 'CHIPRC'
    output_path = '../training_code/Datasets/training_data/' + item

    # labels
    if 'PCIE' in path:  # 'HOLD', 'STAN', 'else'
        classList = {
            "HOLD": 0,
            "STAN": 1,
            "else": 2
        }
    elif 'JQ' in path:
        classList = {
            "COMP": 0,
            "STAN": 1,
            "GAP": 2,
            "TOUCH": 3,
            "MISSING": 4,
            "SHIFT": 5,
            "TPD": 6
        }
    elif 'CHIPRC' in path:  # 'PAD', 'STAN', 'BODY', 'COMP', 'MISSING','BIT']
        classList = {
            "PAD": 0,
            "STAN": 1,
            "BODY": 2,
            "COMP": 3,
            "MISSING": 4,
            "BIT": 5
        }
    elif 'SOLDER' in path:  # 'SOLDER','SOLDER2', 'BIT', 'STAN'
        classList = {
            "SOLDER": 0,
            "SOLDER2": 1,
            "BIT": 2,
            "STAN": 3
        }

    # Variables
    imgFolder = path + '\\' + 'images'
    xmlFolder = path + '\\' + 'xml'
    extract_to = path + '\\' + 'extract'
    saveYoloPath = path + '\\' + 'txt'
    negFolder = ''

    # ============== Steps ===================
    # 0. extract labeled imgs & check labels (can ignore)
    # 1. trans .xml to .txt
    # 2. split train/test
    # ========================================
    extact_labels.extact_labels_to_imgs(imgFolder, xmlFolder, extract_to)
    trans_to_yolo.trans_to_yolo_format(
        imgFolder, xmlFolder, saveYoloPath, negFolder, classList)
    folder = split_traintest.split_train_test(
        saveYoloPath, testRatio, path, output_path)
    print('End')

    # modify yolov5*.yaml config
    folder = folder.split('\\')[-1]
    train_path = '../training_code/Datasets/training_data/' + \
        item + '/' + folder + '/images/train/'
    test_path = '../training_code/Datasets/training_data/' + \
        item + '/' + folder + '/images/val/'
    print('train:' + train_path)
    print('test:' + test_path)
    print('number of classes:' + str(len(classList.keys())))

    data_yaml_path = '../training_code/yolov5/data/' + item + '.yaml'
    with open(data_yaml_path, 'w') as f:
        f.writelines('names: ' + str(list(classList.keys())) + '\n')
        f.writelines('nc: ' + str(len(classList.keys())) + '\n')
        f.writelines(
            'train: ' + os.path.dirname(os.getcwd()) +
            '/training_code/Datasets/training_data/' +
            item + '/' + folder + '/images/train/' + '\n'
        )
        f.writelines(
            'val: ' + os.path.dirname(os.getcwd()) +
            '/training_code/Datasets/training_data/' +
            item + '/' + folder + '/images/val/' + '\n'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path'
    )
    args = parser.parse_args()

    # setting path & testRatio
    # Example:
    #   path = '../03_Training_code/Datasets/raw_data/CHIPRC/2022-06-21_JQ_CHIP_RC'
    #   testRatio = 0.2
    yolo_preprocess(args.path)
