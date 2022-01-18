import numpy as np
import torch
import numpy as np
import torchvision.transforms as transforms
def createbbox(point_list, width=50, hight=50):
    # point_list: [[x1,y1],[x2,y2]]
    # note: YOLO use normalized coordinates
    # return [[x1,y1,w,h],[x2,y2,w,h]]
    bbox = []
    for point in point_list:
        bbox.append([0,point[0]/512,point[1]/512,width/512,hight/512])
    return np.array(bbox).reshape(-1,5)

class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.5,),(0.5,))()(img)
        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets

DEFAULT_TRANSFORMS = transforms.Compose([
    ToTensor(),
])