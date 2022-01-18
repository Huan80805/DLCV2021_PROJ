import os
import numpy as np
import json
from torch.utils.data import Dataset
from PIL import Image
class skullDataset(Dataset):
    def __init__(self, json_path='../skull/records_train.json', train_img_dir='../skull/train',balance_weight=5,transform=None):
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
        yes_series = 0
        #class_weight = [1/27739,1/4926]
        #sample [1:5,0:1]
        for i, series in enumerate(series_list):
            yes_count = 0
            self.seq_len.append(len(series_list[series]))
            if series_list[series][0]['label'] == 0:
                self.series_label.append(0)
            else: self.series_label.append(1)
            for slice in series_list[series]:
                if slice['label'] == 1:  
                    if balance_weight is None:
                        self.path.append(os.path.join(train_img_dir,slice["path"]))
                        self.label.append(1)
                        yes_count += 1
                    else:
                        for i in range(balance_weight):
                            self.path.append(os.path.join(train_img_dir,slice["path"]))
                            self.label.append(1)
                else:
                    self.path.append(os.path.join(train_img_dir,slice["path"]))
                    self.label.append(0)
                    no_count_total += 1
            if yes_count != 0:
                yes_series += 1
                print(yes_count/len(series_list[series]))
            yes_count_total += yes_count
        print(yes_series)
        print(yes_count_total, no_count_total)
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
    print(len(dataset.series_label))
