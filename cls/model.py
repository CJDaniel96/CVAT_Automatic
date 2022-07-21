from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models
from torchvision import transforms as trns
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image


class train_model:
    _defaults = {'model_save' : 'MobileNet_v2',
                }

    def get_mean_std(self, data_dir, batch_size):
        data_transforms = {
                    # 訓練資料集採用資料增強與標準化轉換
                    'train': trns.Compose([
                            trns.Resize((224, 224)), 
                            trns.ToTensor()])}

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                        for x in ['train']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                    shuffle=True)
                    for x in ['train']}

        mean = 0.
        std = 0.
        nb_samples = 0.
        for data, _ in dataloaders['train']:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples
        mean /= nb_samples
        std /= nb_samples
        print('mean:',mean)
        print('std:',std)
        return mean, std

    def data_transform(self, mean, std):
        self.data_transforms = {
            # 訓練資料集採用資料增強與標準化轉換
            'train': trns.Compose([
                    trns.Resize((224, 224)), 
                    trns.ToTensor(), 
                    trns.Normalize(
                    mean=mean, 
                    std=std)]),
            # 驗證資料集僅採用資料標準化轉換
            'val': trns.Compose([
                    trns.Resize((224, 224)), 
                    trns.ToTensor(), 
                    trns.Normalize(
                    mean=mean, 
                    std=std)]),
        }

    def creat_dataset(self, data_dir, batch_size=48):
        # 建立 Dataset
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), self.data_transforms[x])
                        for x in ['train', 'val']}

        # 建立 DataLoader
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size,
                    shuffle=True)
                    for x in ['train', 'val']}
    
    def check_info(self):
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        print(self.dataset_sizes)
        self.class_names = self.image_datasets['train'].classes
        print(self.class_names)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('use ',self.device)

    def tensor2img(self, inp, mean, std):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array(mean)
        std = np.array(std)
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp

    def demo_img(self, mean, std):
        # 取得一個 batch 的訓練資料
        inputs, classes = next(iter(self.dataloaders['train']))
        # 將多張圖片拼成一張 grid 圖
        out = torchvision.utils.make_grid(inputs)
        # 顯示圖片
        img = self.tensor2img(out, mean, std)
        plt.imshow(img)
        plt.title([self.class_names[x] for x in classes])

    def train_model(self, model, num_epochs, model_save_path):
        since = time.time() # 記錄開始時間

        # 記錄最佳模型
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        # 訓練模型主迴圈
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # 對於每個 epoch，分別進行訓練模型與驗證模型
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 將模型設定為訓練模式
                else:
                    model.eval()   # 將模型設定為驗證模式

                running_loss = 0.0
                running_corrects = 0
                # 以 DataLoader 載入 batch 資料
                for inputs, labels in tqdm(self.dataloaders[phase]):
                    # 將資料放置於 GPU 或 CPU
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # 重設參數梯度（gradient）
                    self.optimizer.zero_grad()

                    # 只在訓練模式計算參數梯度
                    with torch.set_grad_enabled(phase == 'train'):
                        # 正向傳播（forward）
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()  # 反向傳播（backward）
                            self.optimizer.step() # 更新參數

                    # 計算統計值
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    # 更新 scheduler
                    self.exp_lr_scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                print('//////////////////////////////////////////')
                print(phase)
                if phase == 'val':
                    save_name = model_save_path.format(epoch,time.strftime('%m%d-%H%M'),'{:.4f}'.format(epoch_acc))
                    print(save_name)
                    # 保存模型
                    torch.save(model, save_name)
                    # 保存權重
                    # torch.save(model.state_dict(), model_save_path.format(time.strftime('%Y%m%d-%H%M'),epoch))

        # 計算耗費時間
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        return model

    def load_model_mobileNet(self):
        # 載入 VGG16 預訓練模型
        self.model_ft = models.mobilenet_v2(pretrained=True)
        # 鎖定 VGG 預訓練模型參數
        if False:
            for param in model_ft.parameters():
                param.requires_grad = False
        # 取得 VGG 最後一層的輸入特徵數量
        # 將 VGG 的最後一層改為只有兩個輸出線性層
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, len(class_names))
        fc_features = self.model_ft.classifier[1].in_features
        self.model_ft.classifier[1] = nn.Linear(fc_features, len(self.class_names))
        # 將模型放置於 GPU 或 CPU
        self.model_ft = self.model_ft.to(self.device)
        # 使用 cross entropy loss
        self.criterion = nn.CrossEntropyLoss()

        # 學習優化器
        self.optimizer = optim.SGD(self.model_ft.parameters(), lr=0.001, momentum=0.9)

        # 每 7 個 epochs 將 learning rate 降為原本的 0.1 倍
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def load_model_VGG(self):
        # 載入 VGG16 預訓練模型
        self.model_ft = models.vgg16_bn(pretrained=True,progress=True)
        # 鎖定 VGG 預訓練模型參數
        if False:
            for param in model_ft.parameters():
                param.requires_grad = False
        # 取得 VGG 最後一層的輸入特徵數量
        # 將 VGG 的最後一層改為只有兩個輸出線性層
        # model_ft.fc = nn.Linear(num_ftrs, len(class_names))
        fc_features = self.model_ft.classifier[6].in_features
        self.model_ft.classifier[6] = nn.Linear(fc_features, len(self.class_names))
        # 將模型放置於 GPU 或 CPU
        model_ft = self.model_ft.to(self.device)
        # 使用 cross entropy loss
        self.criterion = nn.CrossEntropyLoss()
        # 學習優化器
        self.optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # 每 7 個 epochs 將 learning rate 降為原本的 0.1 倍
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

class Inference_model:

    def Inference(self, model_path, image_path, class_names, mean, std):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model_ft = torch.load(model_path, map_location=device)
        
        transforms = trns.Compose([
                    trns.Resize((224, 224)), 
                    trns.ToTensor(), 
                    trns.Normalize(
                    mean=mean, 
                    std=std)])

        # Read image and run prepro
        image = Image.open(image_path).convert("RGB")
        image_tensor = transforms(image)
        image_tensor = image_tensor.unsqueeze(0)
        image = DataLoader(
            image_tensor,
            batch_size=1,
            shuffle=True
        )
        for inputs in tqdm(image):
            print(inputs.size())
            inputs = inputs.to(device)
            outputs = model_ft(inputs)
            print(outputs)
            p, preds = torch.max(outputs, 1)
            print('Predict:',class_names[preds[0]],', value:',float(p[0]))
