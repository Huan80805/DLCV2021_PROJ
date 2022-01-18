

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from torch.nn.modules.pooling import AvgPool2d
from torchsummary import summary
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import nms
import argparse

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputpath', help='input image path', type=str)
parser.add_argument('-o', '--outpath', help='csv output path', type=str)

args = parser.parse_args()
MAXLENGTH = 47


class SkullDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode='test'):
        self.path = path
        self.mode = mode
        if mode == 'train':
            with open(os.path.join(path, 'records_train.json')) as f:
                self.labels = json.load(f)['datainfo']
            self.label = []
            self.mask = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
        )
        self.study_list = sorted(os.listdir(os.path.join(path)))#[low:high]
        self.data = []
        for case_id in tqdm(self.study_list):
            imgs = []
            masks = []
            labels = []
            for slice_file in os.listdir(os.path.join(self.path, case_id)):
                img = np.load(os.path.join(
                    self.path, case_id, slice_file))
                img = np.clip((img+1024)/4095, 0, 255)
                img = Image.fromarray(img)
                img = self.transform(img)
                imgs.append(img)
                if mode == 'train':
                    mask = torch.zeros(1, 512, 512)
                    for coor in self.labels[slice_file[:-4]]['coords']:
                        margin = 10
                        for i in range(coor[0]-margin, coor[0]+margin+1):
                            for j in range(coor[1]-margin, coor[1]+margin+1):
                                try:
                                    mask[0, i, j] = 1
                                except:
                                    continue
                    masks.append(mask)
                    labels.append(self.labels[slice_file[:-4]]['label'])
            for i in range(47-len(imgs)):
                imgs.append(-torch.ones(1, 512, 512))
                if mode == 'train':
                    masks.append(torch.zeros(1, 512, 512))
                    if labels[0] == 0:
                        labels.append(0)
                    else:
                        labels.append(-1)
            self.data.append(torch.stack(imgs, dim=0))
            if mode == 'train':
                self.label.append(torch.tensor(labels, dtype=torch.float32))
                self.mask.append(torch.cat(masks, dim=0))

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.data[idx], self.label[idx], self.mask[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.study_list)


# %%


class PositionalEmbedding1D(nn.Module):

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        return x + self.pos_embedding


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.dc = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.dc(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 4, 4)
        self.dc = DoubleConv(in_ch, out_ch)

    def forward(self, lower, upper):
        lower = self.up(lower)
        out = self.dc(torch.cat([lower, upper], dim=1))
        return out


class Skull_Unet_Transformer(nn.Module):
    def __init__(self, in_ch, depth, trans_layers):
        super().__init__()
        d_model = in_ch*64
        num_class = 2
        self.trans_layers = trans_layers
        self.down_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        self.down_list.append(DoubleConv(1, in_ch))
        ratio = 2
        for i in range(depth):
            self.down_list.append(nn.Sequential(
                nn.MaxPool2d(4),
                DoubleConv(in_ch, in_ch*ratio)
            ))
            in_ch *= ratio
        for i in range(depth):
            self.up_list.append(Up(in_ch, in_ch//ratio))
            in_ch //= ratio
        self.up_list.append(nn.Conv2d(in_ch, num_class, 1))
        trans_encode_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model, dropout=0.1, batch_first=True)
        self.InterTransformer = nn.TransformerEncoder(
            trans_encode_layer, num_layers=trans_layers)
        self.inter_pos_embedding = PositionalEmbedding1D(
            48, d_model)
        self.diagnosis_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.diagnosis_fc = nn.Linear(d_model, 1)
        self.slice_diagnosis_fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        temp = x.reshape(x.shape[0]*x.shape[1], 1, 512, 512)
        out_x = []
        for down_layer in self.down_list:
            temp = down_layer(temp)
            out_x.append(temp)
            # b*l c w h
        # b*l c d_model -> b*l d_model
        transformer_out = out_x[-1].reshape(x.shape[0], x.shape[1], -1)
        transformer_out = self.InterTransformer(self.inter_pos_embedding(
            torch.cat([self.diagnosis_token, transformer_out], dim=1)))
        diagnosis = self.diagnosis_fc(transformer_out[:, 0, :])
        diagnosis = self.sigmoid(diagnosis)
        slice_diagnosis = self.slice_diagnosis_fc(
            transformer_out[:, 1:, :]) # b l 1
        slice_diagnosis = self.sigmoid(slice_diagnosis)
        # b l c w h
        out_x[-1] = transformer_out[:, 1:,
                                    :].reshape(out_x[-1].shape)  # b*l c w h
        for i, up_layer in enumerate(self.up_list):
            if i != (len(self.up_list)-1):
                out_x[-1] = up_layer(out_x[-1], out_x[-i-2])
            else:
                out_x[-1] = up_layer(out_x[-1])

        return diagnosis, slice_diagnosis, out_x[-1].reshape(x.shape[0], MAXLENGTH, 2, 512, 512).permute(0, 2, 1, 3, 4)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=torch.tensor([0.01, 0.99]).cuda(), gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):

        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# %%

test_data = SkullDataset(args.inputpath)

# %%
batchsize = 1
test_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size=batchsize, shuffle=False)
models = [Skull_Unet_Transformer(8, 4, 2).cuda() for _ in range(4)]
name = ['300','600','900','1200']
for i, m in enumerate(models):
    m.load_state_dict(
        torch.load('./model_'+name[i]+'.bin'))
    m.eval()
# %%
sliding_width=11
test = tqdm(test_dataloader)
with open('./transunet_1.csv', 'w') as file:
    file.write('id,label,coords\n')
    for idx, img in enumerate(test):
        fname = os.listdir(os.path.join(
            test_data.path, test_data.study_list[idx]))
        with torch.no_grad():
            diagnosis = 0
            slice_diagnosis = 0
            fracture_mask = 0
            with torch.no_grad():
                for m in models:
                    d, s, f = m(img.cuda())
                    diagnosis += d/len(models)
                    slice_diagnosis += s/len(models)
                    fracture_mask += f/len(models)
            case_diagnosis = torch.round(diagnosis.detach().cpu())
            slice_diagnosis = torch.round(
                slice_diagnosis.squeeze().detach().cpu())
            f_mask = torch.max(fracture_mask, 1)[1][0]
        for slice_num in range(len(fname)):
            slice_diag = int(slice_diagnosis[slice_num])

            if case_diagnosis:
                if slice_diag == 0:

                    slice_diag = -1
                else:
                    c_f_mask=torch.tensor(f_mask[slice_num].unsqueeze(0),dtype=torch.float32)
                    coords = np.argwhere(c_f_mask[0].cpu().numpy() == 1)
                    
                    coords = torch.tensor(coords)
                    boxes = torch.stack([coords[:, 0]-sliding_width//2, coords[:, 1]-sliding_width//2,
                                    coords[:, 0]+sliding_width//2, coords[:, 1]+sliding_width//2],dim=1)
                    if boxes.shape[0]==0:
                        slice_diag = -1
            else:
                slice_diag = 0
            file.write(f'{fname[slice_num][:-4]},{slice_diag},')
            if slice_diag == 1:
                
                reduced_boxes = nms(
                    boxes.detach().cpu().float(),c_f_mask[0,coords[:,0],coords[:,1]].detach().cpu(), 0.5)
                
                if len(reduced_boxes>10):
                    reduced_boxes=reduced_boxes[:10]
                for boxid in reduced_boxes.numpy():
                    file.write(f'{boxes[boxid][0]} {boxes[boxid][1]} ')
            file.write('\n')

import torch.nn as nn
import torch
class AlexNet(nn.Module):
    def __init__(self, num_classes=2, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = AlexNet()
    test_img = torch.rand((1,1,512,512))
    out = model(test_img)

import os
import numpy as np
import json
from torch.utils.data import Dataset
from PIL import Image
'''class skullDataset(Dataset):
    def __init__(self, json_path='/content/skull/records_train.json', train_img_dir='/content/skull/train',balance_weight=5,transform=None):
        with open(json_path) as f:
            records = json.load(f)
        datainfo = records["datainfo"]
        series_list = dict()
        for info in datainfo:
            series_name, slice = '_'.join(info.split('_')[:-1]), info.split('_')[-1]
            if series_name not in series_list:
                series_list[series_name] =  []
            series_list[series_name].append(datainfo[info])
        self.label = []
        self.path = []
        no_count_total = 0
        yes_count_total = 0
        self.seq_len = []
        self.series_label = []
        # yes_series = 0
        #class_weight = [1/27739,1/4926]
        #sample [1:5,0:1]
        for i, series in enumerate(series_list):
            #yes_count = 0
            self.seq_len.append(len(series_list[series]))
            if series_list[series][0]['label'] == 0:
                self.series_label.append(0)
            else: self.series_label.append(1)
            for slice in series_list[series]:
                if slice['label'] == 1:  
                    if balance_weight is None:
                        self.path.append(os.path.join(train_img_dir,slice["path"]))
                        self.label.append(1)
                        #yes_count += 1
                    else:
                        for i in range(balance_weight):
                            self.path.append(os.path.join(train_img_dir,slice["path"]))
                            self.label.append(1)
                else:
                    self.path.append(os.path.join(train_img_dir,slice["path"]))
                    self.label.append(0)
                    #no_count_total += 1
        #     if yes_count != 0:
        #         yes_series += 1
        #         print(yes_count/len(series_list[series]))
            # yes_count_total += yes_count
        # print(yes_series)
        # print(yes_count_total, no_count_total)
        self.transform = transform
        
    def __getitem__(self, index):
        #get 1 series
        path = self.path[index]
        label = self.label[index]
        img = np.load(path)
        img[img>=2550] = 2550
        img[img<0] = 0
        img = img/2550
        img = Image.fromarray(img)
        img = self.transform(img)
        return img.float(), label

    def __len__(self):
        return len(self.label)





if __name__ == '__main__':
    dataset = skullDataset()
    print(len(dataset))#52639
    print(len(dataset.seq_len))
    print(len(dataset.series_label))'''

import torch
def save_checkpoint(checkpoint_path, model):
    torch.save(model.state_dict(), checkpoint_path)
    print('--------model saved to %s-------' % checkpoint_path)
class TrainConfig():
    def __init__(self):
        self.seed = 7
        self.img_size = 512
        self.lr = 5e-5
        self.weight_decay = 1e-3
        self.lr_decay = 0.95

from imghdr import tests
import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
#from model import AlexNet
#from dataset import skullDataset
#from config import TrainConfig
from torchvision import transforms
#from utils import *
from PIL import Image
import csv
class Testset(torch.utils.data.Dataset):
    def __init__(self, test_img_dir, transform):
        self.transform = transform
        study_list = sorted(os.listdir(test_img_dir))
        print(study_list)
        self.path = []
        self.seq_len = []
        for case_id in study_list:
            root = os.listdir(os.path.join(test_img_dir, case_id))
            self.seq_len.append(len(root)) 
            for slice_id in sorted(root):
                self.path.append(os.path.join(test_img_dir, case_id, slice_id))
        print(len(self.path))

    def __getitem__(self, idx):
        path = self.path[idx]
        img = np.load(path)
        img[img>=2550] = 2550
        img[img<0] = 0
        img = img/2550
        img = Image.fromarray(img)
        img = self.transform(img)
        return img.float()

    def __len__(self):
        return len(self.path)
#---------environment setting---------
SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = TrainConfig()
batch_size = 64
print("using device: ", device)
#-------prepare data----
transform = transforms.Compose([
    transforms.ToTensor(),
])
testset = Testset( 
                test_img_dir=args.inputpath,
                transform=transform)
test_dataloader = DataLoader(
    testset,
    batch_size=64,
    num_workers=4,
    shuffle=False)
print("trainset size(# slices):", len(testset))
# build model
model = AlexNet().to(device)
ckpt = torch.load("./alex_finetune.pth")
model.load_state_dict(ckpt)
model.eval()
series_offset = 0
step = 0
log_interval = 40
with torch.no_grad():
    step = 0
    predictions = []
    for batch_idx, images in enumerate(test_dataloader):
        step += 1
        images = images.to(device)
        out = model(images)
        pred = out.max(1, keepdim=True)[1]
        predictions.extend(pred.squeeze().tolist())
        if (step+1)%log_interval == 0:
            print('[Test] {}/{} ({:.0f}%)'.format(
                batch_idx * batch_size, 
                len(testset),
                100.*batch_idx*batch_size/len(testset),
                ))
        step += 1
metric = 'exists one'
n = 2
offset = 0
ratio_thres = 0.04
slices_pred = []
for seq_len in testset.seq_len:
    pred = predictions[offset:offset+seq_len]
    print(pred)
    offset += seq_len
    case_pred = 0
    if metric == 'exists one':
        if(any(pred[i]==1 for i in range(len(pred)))): case_pred=-1
    elif metric == 'consecutive 2':
        if any(pred[i]==1 and pred[i+1]==1 for i in range(len(pred)-1)): case_pred=-1
    elif metric == 'ratio n':
        ratio = sum(pred)/seq_len
        if ratio > ratio_thres:
            case_pred = -1
    slices_pred.extend([case_pred]*seq_len)
print(len(slices_pred))
print(len(testset))
corr = 0
header = ['id','label','coords']
with open('alex_1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for i, row in enumerate(slices_pred):
        writer.writerow([os.path.split(testset.path[i])[-1].split('.')[0],row])

model = AlexNet().to(device)
ckpt = torch.load("./alex_acc0823.pth")
model.load_state_dict(ckpt)
model.eval()
series_offset = 0
step = 0
log_interval = 40
with torch.no_grad():
    step = 0
    predictions = []
    for batch_idx, images in enumerate(test_dataloader):
        step += 1
        images = images.to(device)
        out = model(images)
        pred = out.max(1, keepdim=True)[1]
        predictions.extend(pred.squeeze().tolist())
        if (step+1)%log_interval == 0:
            print('[Test] {}/{} ({:.0f}%)'.format(
                batch_idx * batch_size, 
                len(testset),
                100.*batch_idx*batch_size/len(testset),
                ))
        step += 1
metric = 'exists one'
n = 2
offset = 0
ratio_thres = 0.04
slices_pred = []
for seq_len in testset.seq_len:
    pred = predictions[offset:offset+seq_len]
    print(pred)
    offset += seq_len
    case_pred = 0
    if metric == 'exists one':
        if(any(pred[i]==1 for i in range(len(pred)))): case_pred=-1
    elif metric == 'consecutive 2':
        if any(pred[i]==1 and pred[i+1]==1 for i in range(len(pred)-1)): case_pred=-1
    elif metric == 'ratio n':
        ratio = sum(pred)/seq_len
        if ratio > ratio_thres:
            case_pred = -1
    slices_pred.extend([case_pred]*seq_len)
print(len(slices_pred))
print(len(testset))
corr = 0
header = ['id','label','coords']
with open('alex_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for i, row in enumerate(slices_pred):
        writer.writerow([os.path.split(testset.path[i])[-1].split('.')[0],row])

#!wget https://www.dropbox.com/s/pkyj4qci7wdd47a/823.pth?dl=1 -O 823.pth

import os
import csv
import pandas as pd
import numpy as np
from scipy import stats
result_list = os.listdir('./')
result_list
p1 = pd.read_csv("./transunet_1.csv")
p2 = pd.read_csv("./alex_1.csv")
p3 = pd.read_csv("./alex_2.csv")
p1_class = p1['label']
p2_class = p2['label']
p3_class = p3['label']
p1_name = p1['id']
p2_name = p2['id']
p3_name = p3['id']
new_class = []
new_id = []
def get_key (dict, value):
  return [k for k, v in dict.items() if v == value]
for i in range(len(p1_class)):
    nums = [p1_class[i],p2_class[i],p3_class[i]]
    counts = {}
    for j in nums:
      if(j in counts):
        counts[j] = counts[j]+1
      else:
        counts[j]=1
    
    max_values=max(counts.values())
    max_keys=get_key(counts,max_values)

    print(max_keys[0])

    new_class.append(max_keys[0])
    new_id.append(p2_name[i])
dataframe = pd.DataFrame({'Id':new_id,'label':new_class})
dataframe.to_csv("./voting.csv",index=False,sep=',')

test = pd.read_csv("./voting.csv")
test
