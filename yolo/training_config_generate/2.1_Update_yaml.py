

# %%
import time 
# 只需修改影像路徑(intput_path)
input_path = 'Datasets/training_data/SOLDER/2022-03-14_SOLDER'


# path = '../03_Training_code/' + input_path
local_time = time.localtime() # 取得時間元組
timeString = time.strftime("%Y-%m-%d_", local_time) # 轉成想要的字串形式
item = 'PCIE' if 'PCIE' in input_path else 'SOLDER' if 'SOLDER' in input_path else 'CHIPRC'

# 輸出 data.yaml
data_path = '../03_Training_code/yolov5/data/'+ timeString + item + '.yaml'
print('output data yaml:',data_path)
f = open(data_path, 'w')
f.write('train: ../' + input_path + '/images/train\n')
f.write('val: ../' + input_path + '/images/val\n')
if 'PCIE' in input_path:
    f.write('nc: 3\n')
    f.write('names: '+str(['HOLD', 'STAN', 'else'])+'\n')
elif 'CHIPRC' in input_path:
    f.write('nc: 6\n')
    f.write('names: '+str(['PAD', 'STAN', 'BODY', 'COMP', 'MISSING','BIT'])+'\n')
elif 'SOLDER' in input_path:
    f.write('nc: 4\n')
    f.write('names: '+str(['SOLDER','SOLDER2', 'BIT', 'STAN'])+'\n')
f.close()

# 輸出 models.yaml
models_path = '../03_Training_code/yolov5/models/'+ timeString + item + '.yaml'
print('output models yaml:',models_path)
f = open(models_path, 'w')
f.write('# YOLOv5  by Ultralytics, GPL-3.0 license\n')
if 'PCIE' in input_path:
    f.write('nc: 3  # number of classes\n')
elif 'CHIPRC' in input_path:
    f.write('nc: 6  # number of classes\n')
elif 'SOLDER' in input_path:
    f.write('nc: 4  # number of classes\n')
f.write('# Parameters\n')
f.write('depth_multiple: 0.33  # model depth multiple\n')
f.write('width_multiple: 0.50  # layer channel multiple\n')
f.write('anchors:\n')
f.write('  - [10,13, 16,30, 33,23]  # P3/8\n')
f.write('  - [30,61, 62,45, 59,119]  # P4/16\n')
f.write('  - [116,90, 156,198, 373,326]  # P5/32\n')
f.write('\n')
f.write('# YOLOv5 v6.0 backbone\n')
f.write('backbone:\n')
f.write('  # [from, number, module, args]\n')
f.write('  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2\n')
f.write('   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4\n')
f.write('   [-1, 3, C3, [128]],\n')
f.write('   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8\n')
f.write('   [-1, 6, C3, [256]],\n')
f.write('   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16\n')
f.write('   [-1, 9, C3, [512]],\n')
f.write('   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32\n')
f.write('   [-1, 3, C3, [1024]],\n')
f.write('   [-1, 1, SPPF, [1024, 5]],  # 9\n')
f.write('  ]\n')
f.write('\n')
f.write('# YOLOv5 v6.0 head\n')
f.write('head:\n')
f.write('  [[-1, 1, Conv, [512, 1, 1]],\n')
f.write("   [-1, 1, nn.Upsample, [None, 2, 'nearest']],\n")
f.write('   [[-1, 6], 1, Concat, [1]],  # cat backbone P4\n')
f.write('   [-1, 3, C3, [512, False]],  # 13\n')
f.write('\n')
f.write('   [-1, 1, Conv, [256, 1, 1]],\n')
f.write("   [-1, 1, nn.Upsample, [None, 2, 'nearest']],\n")
f.write('   [[-1, 4], 1, Concat, [1]],  # cat backbone P3\n')
f.write('   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)\n')
f.write('\n')
f.write('   [-1, 1, Conv, [256, 3, 2]],\n')
f.write('   [[-1, 14], 1, Concat, [1]],  # cat head P4\n')
f.write('   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)\n')
f.write('\n')
f.write('   [-1, 1, Conv, [512, 3, 2]],\n')
f.write('   [[-1, 10], 1, Concat, [1]],  # cat head P5\n')
f.write('   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)\n')
f.write('\n')
f.write('   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)\n')
f.write('  ]\n')
f.close()

#%% training code
batch = '8'
epochs = '300'
data = './data/' + timeString + item + '.yaml ' 
cfg = './models/' + timeString + item + '.yaml ' 
weights = './yolov5s.pt ' 
name = item + '/' + timeString.replace('_','')
# hyp = ./data/hyps/hyp.degrees90.yaml

training_code = 'python train.py --batch ' + batch + ' --epochs ' + epochs \
               + ' --data ' + data + '--cfg ' + cfg + '--weights ' + weights \
               + '--name ' + name
print(training_code)

output_txt = 'training_history/' + timeString + item + '.txt'
f = open(output_txt, 'w')
f.write(training_code)
f.close()
# %%

