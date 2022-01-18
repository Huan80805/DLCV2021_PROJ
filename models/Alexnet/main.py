import argparse
from train import train
import os
parser = argparse.ArgumentParser(description="Skull fracture detector")
# Training
parser.add_argument('--resume_ckpt_path', default='best.pth', type=str,
                    help="ckpt to resume training")
parser.add_argument('--log_interval', default=64, type=int,
                    help="validation frequency (by steps)")
parser.add_argument('--save_interval', default=640, type=int,
                    help="validation frequency (by steps)")
parser.add_argument('--epochs', default=50, type=int,
                    help="small for experiment")                    
# Environment
parser.add_argument('--num_workers', default=4, type=int,
                    help="setting num_workers for dataloader")
parser.add_argument('--batch_size', default=64, type=int)
# Path
parser.add_argument('--reset', action='store_true',help='reset ckpt dir')
parser.add_argument('--ckpt_dir', default='ckpt/exp', type=str,
                    help="saved_ckpt_dir")
parser.add_argument('--train_img_dir', default='../skull/train', type=str,
                    help="Training images directory")
parser.add_argument('--json_path', default='../skull/records_train.json', type=str)

args = parser.parse_args()
if args.reset:
    os.system('rm -rf ' + args.ckpt_dir)
os.makedirs(args.ckpt_dir, exist_ok=True)
train(args)