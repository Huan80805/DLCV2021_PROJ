from email.mime import image
import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
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
print("using device: ", device)
def train(args):
    #-------exp--------
    #--------log-------
    wandb.init("dlcv_proj", config=config)
    wandb.config.update(args)
    wandb.config.update({'device':device})
    #-------prepare data----
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # dataset = skullDataset(json_path=args.json_path, 
    #                 train_img_dir=args.train_img_dir,
    #                 balance_weight=5,
    #                 transform=transform)
    # trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
    dataset = skullDataset(json_path=args.json_path, 
                    train_img_dir=args.train_img_dir,
                    balance_weight=1,
                    transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
    train_dataloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)
    val_dataloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False)
    print("trainset size(# slices):", len(trainset))
    # build model
    model = AlexNet().to(device)
    #finetune
    optim = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, config.lr_decay)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.
    step = 0
    train_loss = 0.
    train_acc = 0.
    for epoch in range(args.epochs):
        # -----------Train----------
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            model.train()
            model.zero_grad()
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            pred = out.max(1, keepdim=True)[1]
            train_loss += loss.item()
            train_acc += pred.eq(labels.view_as(pred)).float().mean().item()
            optim.step()
            step += 1
            # Statistic
            if (step+1) % args.log_interval == 0 :
                train_loss /= args.log_interval
                train_acc = train_acc/args.log_interval * 100.
                print('[Train] Epoch: {} [{}/{} ({:.0f}%)] -loss- {:.6f} -acc- {:.2f}%'.format(
                epoch, batch_idx * args.batch_size, 
                len(trainset),
                100.*batch_idx*args.batch_size/len(trainset),
                train_loss,
                train_acc
                ))
                wandb.log({
                    "train_loss":train_loss,
                    "train_acc":train_acc
                })
                train_loss = 0.
                train_acc = 0.
            if (step+1)% args.save_interval ==0:
                val_acc = eval(args, model, val_dataloader, config)    
                if val_acc > best_acc:
                    ckpt_path = os.path.join(args.ckpt_dir,"step{}.pth".format(step))
                    save_checkpoint(ckpt_path,model)
                best_acc = val_acc
        scheduler.step()

# -------------Validation--------
def eval(args, model, val_dataloader, config):
    model.eval()
    with torch.no_grad():
        labels = []
        step = 0
        val_loss = 0.
        val_acc = 0
        criterion = nn.CrossEntropyLoss()
        for batch_idx, (images, labels) in enumerate(val_dataloader):
            step += 1
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            val_loss += criterion(out, labels)
            pred = out.max(1, keepdim=True)[1]
            val_acc += pred.eq(labels.view_as(pred)).sum().item()
    val_loss /= len(val_dataloader.dataset)
    val_acc = val_acc /len(val_dataloader.dataset) * 100.
    print('[Validation] -loss- {:.6f} -acc- {:.2f}%'.format(
            val_loss, val_acc))
    wandb.log({
        "val_loss":val_loss,
        "val_acc":val_acc
    })
    return val_acc