import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
from utils import *
class skullDataset(Dataset):
    def __init__(self, json_path='records_train.json', train_img_dir='train',window_size=20):
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
        self.bbox = []
        for i, series in enumerate(series_list):
            for slice in series_list[series]:
                self.path.append(os.path.join(train_img_dir,slice["path"]))
                if slice['label'] == 0:
                    self.label.append(0)
                else:
                    self.label.append(1)
                self.bbox.append(createbbox(slice['coords'],width=window_size,hight=window_size))
        self.transform = DEFAULT_TRANSFORMS

    def __getitem__(self, index):
        #get 1 series
        path = self.path[index]
        img = np.load(path)
        img = Image.fromarray(img)
        img, bbox = self.transform((img, self.bbox[index]))
        return img.float(), self.label[index], bbox

    def __len__(self):
        return len(self.label)

    def collate_fn(self, batch):
        #ignore images with no bboxes
        #add sample index from same slice
        batch = [data for data in batch if data is not None]
        imgs, labels, bboxes = list(zip(*batch))
        imgs = torch.stack([img for img in imgs])
        tgt_bboxes = []
        for idx,slice_bboxes in enumerate(bboxes):
            if slice_bboxes.shape[0] == 0:
                continue
            slice_bboxes[:,0] = idx
            tgt_bboxes.append(slice_bboxes[:,:])
        if len(tgt_bboxes) != 0:
            tgt_bboxes = torch.cat(tgt_bboxes, 0)
        else:
            tgt_bboxes = None
        return imgs, labels, tgt_bboxes



if __name__ == '__main__':
    # with open('records_train.json') as f:
    #     records = json.load(f)
    # datainfo = records["datainfo"]
    # series_list = dict()
    # for info in datainfo:
    #     series_name, slice = '_'.join(info.split('_')[:-1]), info.split('_')[-1]
    #     if series_name not in series_list:
    #         series_list[series_name] =  []
    #     series_list[series_name].append(datainfo[info])
    # print(len(series_list))
    # label = []
    # path = []
    # coords = []
    # for i, series in enumerate(series_list):
    #     path.append([])
    #     label.append([])
    #     coords.append([])
    #     for slice in series_list[series]:
    #         path[i].append(slice["path"])
    #         label[i].append(slice['label'])
    #         coords[i].append(slice['coords'])
    # print(len(path))
    # for i,img_path in enumerate(path[0]):
    #     plt.subplot((len(path[0])+1)//5,5,i+1)
    #     img = np.load(os.path.join('train',img_path))
    #     img = (img+1024)/(1024+4071)*255
    #     plt.imshow(img)
    # plt.savefig("test_skull.png")
    # plt.show()
    # trainset = skullDataset()
    # # print(trainset[-1][0].shape)
    # # print(len(trainset[-1][1]),'\n')
    # # print(trainset[-1][2])
    # dataloader = DataLoader(
    #     trainset,
    #     batch_size=2,
    #     collate_fn=trainset.collate_fn,
    #     shuffle=True)
    # dataloader = iter(dataloader)
    # images, labels, bboxes = next(dataloader)
    # print(bboxes.shape)
    # print(images.shape)
    test_tensor = torch.tensor(1.2).float()
    a = test_tensor.long()
    print(a)
    a = a.float()
    print(a)