from imghdr import tests
import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from model import AlexNet
from dataset import skullDataset
from config import TrainConfig
from torchvision import transforms
from utils import *
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
testset = skullDataset(json_path='../skull/records_train.json', 
                train_img_dir="../skull/train/",
                balance_weight=None,
                transform=transform)
test_dataloader = DataLoader(
    testset,
    batch_size=64,
    num_workers=4,
    shuffle=False)
print("trainset size(# slices):", len(testset))
# build model
model = AlexNet().to(device)
ckpt = torch.load("best.pth")
model.load_state_dict(ckpt)
model.eval()
series_offset = 0
step = 0
log_interval = 40
with torch.no_grad():
    labels = []
    step = 0
    slice_loss = 0.
    slice_acc = 0
    criterion = nn.CrossEntropyLoss()
    predictions = []
    for batch_idx, (images, labels) in enumerate(test_dataloader):
        step += 1
        images, labels= images.to(device), labels.to(device)
        out = model(images)
        slice_loss += criterion(out, labels)
        pred = out.max(1, keepdim=True)[1]
        predictions.extend(pred.squeeze().tolist())
        slice_acc += pred.eq(labels.view_as(pred)).sum().item()
        if (step+1)%log_interval == 0:
            print('[Validation] {}/{} ({:.0f}%)'.format(
                batch_idx * batch_size, 
                len(testset),
                100.*batch_idx*batch_size/len(testset),
                ))
        step += 1
metrics = ['exist', 'consecutive 2', 'ration n']
exist_n = 2
n = 2
offset = 0
ratio_thres = 0.04
for metric in metrics:
    series_pred = []
    for seq_len in testset.seq_len:
        pred = predictions[offset:offset+seq_len]
        offset += seq_len
        case_pred = 0
        if metric == 'exist':
            positive = sum(pred)
            if (positive) >= exist_n:
                case_pred = 1
        elif metric == 'consecutive 2':
            if any(pred[i]==1 and pred[i+1]==1 for i in range(len(pred)-1)): case_pred=1
        elif metric == 'ratio n':
            ratio = sum(pred)/seq_len
            if ratio > ratio_thres:
                case_pred = 1
        series_pred.append(case_pred)
    corr = 0
    for i, label in enumerate(testset.series_label):
        if label == series_pred[i]:
            corr += 1
    print('[Slice] -loss- {:.6f} -acc- {:.2f}%'.format(
            slice_loss/len(testset), slice_acc/len(testset)*100.))
    print('[CASE] -acc- {:.2f} metric:{}'.format(corr/len(series_pred),metric))