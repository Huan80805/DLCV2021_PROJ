#testing yolo in/output
from pytorchyolo import models
import torch
from utils import *
from dataset import skullDataset
import argparse
from torch.utils.data import DataLoader
parser = argparse.ArgumentParser(description="Skull fracture detector")
parser.add_argument('--train_img_dir', default='../skull/train', type=str,
                    help="Training images directory")
parser.add_argument('--json_path', default='../skull/records_train.json', type=str)

args = parser.parse_args()
# Load the YOLO model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device:", device)
model = models.load_model(
    model_path="yolov3-custom.cfg",
    weights_path="step58999.pth"
).to(device)
batch_size = 16
img_size = 512
num_workers = 4
log_interval = 20
valset = skullDataset(json_path=args.json_path, train_img_dir=args.train_img_dir)
val_dataloader = DataLoader(
    valset,
    batch_size=batch_size,
    collate_fn=valset.collate_fn,
    num_workers=num_workers,
    shuffle=False)
torch.set_printoptions(profile="full")
#----------testing-----------
model.eval()
with torch.no_grad():
    sample_metrics = []  # List of tuples (TP, confs, pred)
    labels = []
    step = 0
    fp = 0
    for batch_idx, (images, _, bboxes) in enumerate(val_dataloader):
        step += 1
        images = images.to(device)
        out = model(images)
        conf_thres, iou_thres = 1e-4, 0.6
        out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres)
        if type(bboxes) == type(None):
            out = torch.cat(out,dim=0)
            fp += out.shape[0]
            continue
        #extract class labels
        labels += bboxes[:, 1].tolist()
        #scale target to original image size
        bboxes[:, 2:] = xywh2xyxy(bboxes[:, 2:])
        bboxes[:, 2:] *= img_size
        sample_metrics += get_batch_statistics(out, bboxes, iou_threshold=iou_thres)
        if step % log_interval == 0 :
            print('[Val] [{}/{} ({:.2f}%)]'.format(
            batch_idx * batch_size, 
            len(valset),
            100.*batch_idx*batch_size/len(valset)))
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)
    precision, recall, AP, f1, ap_class = metrics_output
print('[Validation] Precision:{} Recall:{} AP:{} F1:{}'.format(
        precision, recall, AP, f1))