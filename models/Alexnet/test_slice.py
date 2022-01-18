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
                test_img_dir="../skull/test/",
                transform=transform)
test_dataloader = DataLoader(
    testset,
    batch_size=64,
    num_workers=4,
    shuffle=False)
print("trainset size(# slices):", len(testset))
# build model
model = AlexNet().to(device)
ckpt = torch.load("finetune.pth")
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
exist_n = 1
offset = 0
slices_pred = []
for seq_len in testset.seq_len:
    pred = predictions[offset:offset+seq_len]
    offset += seq_len
    case_pred = 0
    positive = sum(pred)
    if (positive) >= exist_n:
        case_pred = 1
    # slices_pred.extend([case_pred]*seq_len)
        for i in range(seq_len):
            if pred[i] == 1:
                slices_pred.append(1)
            else:
                slices_pred.append(-1)
    else:
        slices_pred.extend([0]*seq_len)
print(len(slices_pred))
print(len(testset))
corr = 0
header = ['id','label','coords']
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for i, row in enumerate(slices_pred):
        writer.writerow([os.path.split(testset.path[i])[-1].split('.')[0],row])