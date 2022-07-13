#%%
from model import train_model
model = train_model()

n = 0 # 0:MobilNet v2；1:VGG16
data_dir = r'E:\USI\spyder\MobileNet\JQ_CHIPRC_ORG\datasets\Data_0704' #'chip_split'
model_cls = ['MobileNet_v2', 'VGG16']
model_save = model_cls[n]
model_save_path = './saved_models/save_CHIPRC_ORG_20220704/'+model_save+'_epoch{}_{}_ACC{}.pt'
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
