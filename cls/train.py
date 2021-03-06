import argparse
import os
from cls.model import train_model


def train(data_dir, model_save_dir):
    model = train_model()

    n = 0 # 0:MobilNet v2；1:VGG16

    model_cls = ['MobileNet_v2', 'VGG16']
    model_save = model_cls[n]
    model_save_path = model_save_dir + '_' + model_save + '_epoch{}_{}_ACC{}.pt'
    batch_size = 64 if n == 0 else 8 # mobileNet max:64, VGG max:16
    num_epochs = 40

    mean, std = model.get_mean_std(data_dir, batch_size)
    para_txt = data_dir + r'\mean_std.txt'
    file = open(para_txt, 'w')
    file.write('mean: ' + str(mean) + '\n')
    file.write('std: ' + str(std))
    file.close()

    model.data_transform(mean, std)
    model.creat_dataset(data_dir, batch_size)
    model.check_info()
    model.demo_img(mean, std)
    model.load_model_mobileNet() if n == 0 else model.load_model_VGG()
    model_ft = model.train_model(model.model_ft, num_epochs, model_save_path)

def create_save_dir(model_save_dir):
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
        print('Create ' + model_save_dir + ' Success!')

def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', help=r'ex: E:\USI\spyder\MobileNet\JQ_CHIPRC_ORG\datasets\Data_0704')
    parser.add_argument('--model-save-dir', help='ex: ./saved_models/save_CHIPRC_ORG_20220704/')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argsparser()
    create_save_dir(args.model_save_dir)
    train(args.data_dir, args.model_save_dir)