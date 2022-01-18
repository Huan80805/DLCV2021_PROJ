import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from pytorchyolo import models
from dataset import skullDataset
from config import TrainConfig
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
print("using device: ", device)
def train(args):
    #-------exp--------
    #--------log-------
    wandb.init("dlcv_proj", config=config)
    wandb.config.update(args)
    wandb.config.update({'device':device})
    #-------prepare data----
    dataset = skullDataset(json_path=args.json_path, 
        train_img_dir=args.train_img_dir,
        window_size=config.window_size)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
    train_dataloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        shuffle=True)
    val_dataloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        shuffle=False)
    
    print("trainset size(# slices):", len(trainset))
    # build model
    model = models.load_model("./yolov3-custom.cfg").to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params=params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, config.lr_decay)
    obj_loss = 0.
    iou_loss = 0.
    cls_loss = 0.
    best_f1 = 0.
    step = 0
    for epoch in range(args.epochs):
        # -----------Train----------
        for batch_idx, (images, _, bboxes) in enumerate(train_dataloader):
            model.train()
            model.zero_grad()
            if type(bboxes) == type(None):
                #todo: criterion of no bounding boxes?
                continue
            images, bboxes = images.to(device), bboxes.to(device)
            out = model(images)
            loss, loss_components = compute_loss(out, bboxes, model)
            #loss_components: IOUloss, ObjectLoss, ClassLoss, Loss
            loss.backward()
            optim.step()
            iou_loss += loss_components[0].item()
            obj_loss += loss_components[1].item()
            cls_loss += loss_components[2].item()
            # scheduler.step()
            # Statistic
            if (step+1) % args.log_interval == 0 :
                obj_loss /= args.log_interval
                iou_loss /= args.log_interval
                cls_loss /= args.log_interval
                print('[Train] Epoch: {} [{}/{} ({:.0f}%)] -loss- IOU:{:.6f},OBJ:{:.6f},CLS:{:.6f}'.format(
                epoch, batch_idx * args.batch_size, 
                len(trainset),
                100.*batch_idx*args.batch_size/len(trainset),
                iou_loss,
                obj_loss,
                cls_loss
                ))
                wandb.log({
                "IOU_LOSS": iou_loss,
                "OBJ_LOSS": obj_loss,
                "CLS_LOSS": cls_loss,
                })
                iou_loss = 0.
                obj_loss = 0.
                cls_loss = 0.
            step += 1
            if (step+1)% args.save_interval ==0:
                val_f1 = eval(args, model, val_dataloader, config)    
                if val_f1 > best_f1:
                    ckpt_path = os.path.join(args.ckpt_dir,"step{}.pth".format(step))
                    save_checkpoint(ckpt_path,model)
                best_f1 = val_f1
        scheduler.step()

# -------------Validation--------
def eval(args, model, val_dataloader, config):
    model.eval()
    with torch.no_grad():
        sample_metrics = []  # List of tuples (TP, confs, pred)
        labels = []
        step = 0
        for batch_idx, (images, _, bboxes) in enumerate(val_dataloader):
            if type(bboxes) == type(None):
                continue
            step += 1
            images = images.to(device)
            #extract class labels
            labels += bboxes[:, 1].tolist()
            #scale target to original image size
            bboxes[:, 2:] = xywh2xyxy(bboxes[:, 2:])
            bboxes[:, 2:] *= config.img_size
            out = model(images)
            #for fast training use 1e-2
            #for acc  testing use 1e-3
            conf_thres, iou_thres = 1e-2, 0.45
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres)
            sample_metrics += get_batch_statistics(out, bboxes, iou_threshold=iou_thres)
            if step % args.log_interval == 0 :
                print('[Val] [{}/{} ({:.2f}%)]'.format(
                batch_idx * args.batch_size, 
                len(val_dataloader.dataset),
                100.*batch_idx*args.batch_size/len(val_dataloader.dataset)))
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_labels, labels)
        precision, recall, AP, f1, ap_class = metrics_output
    print('[Validation] Precision:{} Recall:{} AP:{} F1:{}'.format(
            precision, recall, AP, f1))
    wandb.log({
        "precision":precision,
        "recall":recall,
        "f1":f1,
        "AP":AP
    })
    return f1