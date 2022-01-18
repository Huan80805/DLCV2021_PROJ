# %%
import torchvision
import torch
import numpy as np
import os
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm
from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys
# %%

class SkullDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
        )
        self.study_list = sorted(os.listdir(
            os.path.join(path)))
        self.imgs = []
        for case_id in tqdm(self.study_list):
            for slice_file in os.listdir(os.path.join(self.path, case_id)):
                img = np.load(os.path.join(
                    self.path, case_id, slice_file))
                img = np.clip((img+1024)/4095, 0, 255)
                img = Image.fromarray(img)
                img = self.transform(img)
                self.imgs.append(img)

    def __getitem__(self, idx):
        return self.imgs[idx],None

    def __len__(self):
        return len(self.imgs)


test_data = SkullDataset(sys.argv[1])
def collate_fn(batch):
    return tuple(zip(*batch))
test_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
# %%
device = 'cuda'
model = torchvision.models.detection.fasterrcnn_resnet50_fpn().to(device)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).cuda()
model.load_state_dict(torch.load('./fastrcnn.bin'))
model.eval()
# %%
with open(sys.argv[3], 'w') as file:
    with open(sys.argv[2],'r') as slice_pred:
        file.write('id,label,coords\n')
        for slice_prediction,(test_img,_) in tqdm(zip(slice_pred.readlines()[1:],test_dataloader)):
            #print(slice_prediction)
            if slice_prediction.split(',')[1].strip()=='-1':
                test_img = list(i.cuda() for i in test_img)
                file.write(slice_prediction.strip().split(',')[0])
                file.write(',')

                with torch.no_grad():
                    outputs = model(test_img)
                    boxes=outputs[0]['boxes'].detach()
                    scores=outputs[0]['scores'].detach()
                #boxes_id= nms(boxes+torch.tensor([-5,-5,+5,+5]).cuda(),scores, 0.5).cpu().numpy()
                #boxes=boxes[boxes_id].cpu().numpy()
                #scores=scores[:len(boxes)].cpu().numpy()
                if len(boxes>10):
                    boxes=boxes[:10]
                elif len(boxes)==0 :
                    file.write('-1,\n')
                    continue
                threshold=0.93
                if scores[0]<threshold:
                    file.write('-1,\n')
                    continue
                file.write('1,')
                for box,score in zip(boxes,scores):
                    if score>threshold:
                        file.write(f'{int((box[:][0]+box[:][2])//2)} {int((box[:][1]+box[:][3])//2)} ')
                file.write('\n')
            else:
                file.write(slice_prediction)